use crate::registry::ShardLifecycleState;
use crate::report::{ShardError, StartupReport};
use crate::runtime::Jax;
use crate::shard::Shard;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::error::Error;
use futures::stream::{FuturesUnordered, StreamExt};
use petgraph::prelude::NodeIndex;

use super::scheduler::ShardScheduler;

impl Jax {
    /// Starts all registered shards in dependency-aware topological order with bounded concurrency.
    pub async fn start(self: &Arc<Self>) -> Result<StartupReport, Box<dyn Error + Send + Sync>> {
        let snapshot = {
            let _mutation = self.registry.lock_mutation().await;
            self.begin_start()?;
            match self.registry.snapshot() {
                Some(snapshot) => snapshot,
                None => {
                    self.finish_start(false);
                    return Err("Jax: Registry null".into());
                }
            }
        };

        let scheduler = Arc::new(ShardScheduler::new(&snapshot.graph));
        let mut tasks = FuturesUnordered::new();
        let mut report = StartupReport::default();

        let mut ready_queue = VecDeque::new();
        let max_concurrency = self.config.max_concurrency.max(1);

        let make_task = |shard: Arc<dyn Shard>, idx: NodeIndex| {
            let jax_ptr = Arc::clone(self);
            let scheduler_ptr = Arc::clone(&scheduler);
            let registry = Arc::clone(&snapshot);
            let shard_id = shard.descriptor().id();
            let probes = &self.config.probes;
            async move {
                scheduler_ptr.mark_running(idx);
                registry.set_state(shard_id, ShardLifecycleState::Starting);

                for probe in probes {
                    if let Err(e) = probe.before_setup(shard.as_ref()).await {
                        registry.set_state(shard_id, ShardLifecycleState::Failed);
                        return Err((shard_id, idx, e));
                    }
                }

                let result = shard.setup(jax_ptr).await;

                for probe in probes {
                    probe.after_setup(shard.as_ref(), &result).await;
                }

                match result {
                    Ok(_) => {
                        registry.set_state(shard_id, ShardLifecycleState::Started);
                        Ok((shard_id, idx))
                    }
                    Err(e) => {
                        registry.set_state(shard_id, ShardLifecycleState::Failed);
                        Err((shard_id, idx, e))
                    }
                }
            }
        };

        for item in scheduler.collect_seeds() {
            ready_queue.push_back(item);
        }

        loop {
            while tasks.len() < max_concurrency && !ready_queue.is_empty() {
                let Some((shard, idx)) = ready_queue.pop_front() else {
                    break;
                };
                tasks.push(make_task(shard, idx));
            }

            if tasks.is_empty() {
                break;
            }

            if let Some(res) = tasks.next().await {
                match res {
                    Ok((_id, idx)) => {
                        for next_item in scheduler.notify_success(idx) {
                            ready_queue.push_back(next_item);
                        }
                    }
                    Err((id, idx, err)) => {
                        let skipped_ids = scheduler.notify_failure(idx);
                        for skipped_id in &skipped_ids {
                            snapshot.set_state(*skipped_id, ShardLifecycleState::Skipped);
                        }
                        report.skipped.extend(skipped_ids);
                        report.failed.push(ShardError { id, error: err });
                    }
                }
            }
        }

        self.finish_start(report.is_success());

        Ok(report)
    }
}
