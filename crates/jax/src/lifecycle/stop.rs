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
        let snapshot = match self.registry.snapshot() {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut layers = compute_shard_layers(&snapshot.graph)?;
        layers.reverse();

        let max_concurrency = self.config.max_concurrency;
        let probes = &self.config.probes;
        let mut all_errors: Vec<Box<dyn Error + Send + Sync>> = Vec::new();

        for layer in layers.into_iter() {
            let results = stream::iter(layer)
                .map(|shard| {
                    let jax_ptr = Arc::clone(self);
                    async move {
                        for probe in probes {
                            probe.before_teardown(shard.as_ref()).await;
                        }

                        let result = shard.teardown(jax_ptr).await;

                        for probe in probes {
                            probe.after_teardown(shard.as_ref(), &result).await;
                        }

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

        if let Some(first_err) = all_errors.into_iter().next() {
            Err(first_err)
        } else {
            Ok(())
        }
    }
}
