use crate::registry::ShardLifecycleState;
use crate::runtime::Jax;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::error::Error;
use futures::stream;
use futures::stream::StreamExt;

use super::layers::compute_shard_layers;

impl Jax {
    /// Graceful shutdown of all shards using best-effort reverse topological order.
    pub async fn stop(self: &Arc<Self>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let snapshot = {
            let _mutation = self.registry.lock_mutation().await;
            if !self.begin_stop()? {
                return Ok(());
            }

            match self.registry.snapshot() {
                Some(s) => s,
                None => {
                    self.finish_stop();
                    return Ok(());
                }
            }
        };

        let mut layers = match compute_shard_layers(&snapshot.graph) {
            Ok(layers) => layers,
            Err(error) => {
                self.abort_stop();
                return Err(error);
            }
        };
        layers.reverse();

        let max_concurrency = self.config.max_concurrency.max(1);
        let probes = &self.config.probes;
        let mut all_errors: Vec<Box<dyn Error + Send + Sync>> = Vec::new();

        for layer in layers.into_iter() {
            let layer = layer
                .into_iter()
                .filter(|shard| {
                    snapshot.state(shard.descriptor().id()) == Some(ShardLifecycleState::Started)
                })
                .collect::<Vec<_>>();

            if layer.is_empty() {
                continue;
            }

            let results = stream::iter(layer)
                .map(|shard| {
                    let jax_ptr = Arc::clone(self);
                    let registry = Arc::clone(&snapshot);
                    async move {
                        let shard_id = shard.descriptor().id();
                        registry.set_state(shard_id, ShardLifecycleState::Stopping);

                        for probe in probes {
                            probe.before_teardown(shard.as_ref()).await;
                        }

                        let result = shard.teardown(jax_ptr).await;

                        for probe in probes {
                            probe.after_teardown(shard.as_ref(), &result).await;
                        }

                        registry.set_state(shard_id, ShardLifecycleState::Stopped);
                        result
                    }
                })
                .buffer_unordered(max_concurrency)
                .collect::<Vec<_>>()
                .await;

            for res in results {
                if let Err(e) = res {
                    all_errors.push(e);
                }
            }
        }

        self.finish_stop();

        if let Some(first_err) = all_errors.into_iter().next() {
            Err(first_err)
        } else {
            Ok(())
        }
    }
}
