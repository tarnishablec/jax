use crate::jax::Jax;
use crate::report::{ShardError, StartupReport, StoredStartupReport};
use crate::shard::Shard;
use crate::shard::layer::compute_shard_layers;
use crate::shard::schedule::ShardScheduler;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::error::Error;
use core::sync::atomic::Ordering;
use futures::stream;
use futures::stream::{FuturesUnordered, StreamExt};

impl Jax {
    /// Starts all registered shards in dependency-aware topological order with bounded concurrency.
    pub async fn start(self: &Arc<Self>) -> Result<StartupReport, Box<dyn Error + Send + Sync>> {
        let snapshot = self.registry.snapshot().ok_or("Jax: Registry null")?;

        let scheduler = Arc::new(ShardScheduler::new(&snapshot.graph));
        let mut tasks = FuturesUnordered::new();
        let mut report = StartupReport::default();

        let mut ready_queue = VecDeque::new();
        let max_concurrency = self.config.max_concurrency;

        let make_task = |shard: Arc<dyn Shard>, idx: usize| {
            let jax_ptr = Arc::clone(self);
            let scheduler_ptr = Arc::clone(&scheduler);
            let shard_id = shard.id();
            let probes = &self.config.probes;
            async move {
                scheduler_ptr.mark_running(idx);

                for probe in probes {
                    if let Err(e) = probe.before_setup(shard.as_ref()).await {
                        return Err((shard_id, idx, e));
                    }
                }

                let result = shard.setup(jax_ptr).await;

                for probe in probes {
                    probe.after_setup(shard.as_ref(), &result).await;
                }

                match result {
                    Ok(_) => Ok((shard_id, idx)),
                    Err(e) => Err((shard_id, idx, e)),
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
                        report.skipped.extend(skipped_ids);
                        report.failed.push(ShardError { id, error: err });
                    }
                }
            }
        }

        let stored = StoredStartupReport {
            failed_ids: report.failed.iter().map(|f| f.id).collect(),
            skipped: report.skipped.clone(),
        };
        let stored_ptr = Box::into_raw(Box::new(stored));
        let old_report = self.startup_report.swap(stored_ptr, Ordering::AcqRel);
        if !old_report.is_null() {
            unsafe {
                drop(Box::from_raw(old_report));
            }
        }

        Ok(report)
    }

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
