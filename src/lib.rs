#![no_std]
extern crate alloc;

pub mod config;
pub mod report;
mod shard;

use crate::config::JaxConfig;
use crate::report::{ShardError, StartupReport};
pub use crate::shard::Shard;
use crate::shard::layer::compute_shard_layers;
use crate::shard::schedule::ShardScheduler;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::type_name;
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
use futures::stream;
use futures::stream::{FuturesUnordered, StreamExt};
use petgraph::prelude::*;
use uuid::Uuid;

pub(crate) type ShardGraph = StableDiGraph<Arc<dyn Shard>, ()>;

pub(crate) struct ShardRegistry {
    pub(crate) graph: ShardGraph,
    pub(crate) native_indices: BTreeMap<Uuid, NodeIndex>,
    #[allow(dead_code)]
    pub(crate) guest_indices: BTreeMap<Uuid, NodeIndex>,
}

/// Core application context.
///
/// Lifecycle: `default/with_config()` → `register()` → `Arc::new()` → `start()`.
#[derive(Default)]
pub struct Jax {
    registry: AtomicPtr<ShardRegistry>,
    pending: Vec<Arc<dyn Shard>>,
    //
    config: JaxConfig,
}

impl Jax {
    pub fn with_config(config: JaxConfig) -> Self {
        Self {
            config,
            registry: Default::default(),
            pending: Default::default(),
        }
    }

    /// Register a shard. Must be called before `start()`.
    pub fn register<T: Shard>(mut self, shard: Arc<T>) -> Self {
        self.pending.push(shard);
        self
    }

    /// Finalizes the registry by building the dependency graph.
    /// This is an incremental-rebuild: it loads the current registry, clones its data, and merges the new shards.
    pub fn build(mut self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        if self.pending.is_empty() {
            return Ok(self);
        }

        // 1. Load the existing state or initialize new containers
        let (mut graph, mut indices) = if let Some(snapshot) = self.snapshot_registry() {
            // Clone the existing graph and index map to perform incremental updates
            (snapshot.graph.clone(), snapshot.native_indices.clone())
        } else {
            (StableDiGraph::new(), BTreeMap::new())
        };

        // 2. Drain new shards from the pending queue
        let new_shards: Vec<Arc<dyn Shard>> = self.pending.drain(..).collect();
        let mut new_node_indices = Vec::with_capacity(new_shards.len());

        // 3. Add new shards as nodes
        for shard in new_shards {
            let shard_id = shard.id();

            // Prevent overwriting existing shards
            if indices.contains_key(&shard_id) {
                return Err(
                    alloc::format!("Jax: Shard [{}] already exists in registry", shard_id).into(),
                );
            }

            let idx = graph.add_node(shard);
            indices.insert(shard_id, idx);
            new_node_indices.push(idx);
        }

        // 4. Update edges for the NEW nodes
        // We only need to scan dependencies for the shards we just added
        for &consumer_idx in &new_node_indices {
            let dep_ids = {
                let shard = &graph[consumer_idx];
                shard.dependencies()
            };

            for dep_id in dep_ids {
                if let Some(&provider_idx) = indices.get(&dep_id) {
                    // Direction: Provider -> Consumer
                    graph.add_edge(provider_idx, consumer_idx, ());
                } else {
                    return Err(alloc::format!(
                        "Jax: Missing dependency [{}] for new shard [{}]",
                        dep_id,
                        graph[consumer_idx].id()
                    )
                    .into());
                }
            }
        }

        // 5. Verify graph integrity (Cycle Detection)
        if petgraph::algo::is_cyclic_directed(&graph) {
            return Err("Jax: Incremental build failed due to circular dependency".into());
        }

        // 6. Atomically update the registry
        let new_registry = Arc::into_raw(Arc::new(ShardRegistry {
            graph,
            native_indices: indices,
            guest_indices: Default::default(),
        })) as *mut ShardRegistry;

        let old_ptr = self.registry.swap(new_registry, Ordering::AcqRel);

        // 7. Safe cleanup
        if !old_ptr.is_null() {
            unsafe {
                drop(Arc::from_raw(old_ptr));
            }
        }

        Ok(self)
    }

    /// Starts all registered shards in **dependency-aware topological order** with bounded concurrency.
    ///
    /// ### Startup Strategy Overview
    ///
    /// 1. **Topological scheduling (dynamic Kahn's algorithm variant)**:
    ///    - Shards are started only after **all their dependencies** have successfully completed setup.
    ///    - Uses a **ready queue** + atomic in-degree tracking (via `ShardScheduler`) to dynamically discover
    ///      newly ready shards as upstream dependencies finish.
    ///    - This guarantees the correct initialization order: providers before consumers.
    ///
    /// 2. **Bounded concurrency**:
    ///    - At most `self.config.max_concurrency` shards run their `setup()` simultaneously.
    ///    - Prevents overwhelming the system (e.g., too many network connections, database sessions,
    ///      or CPU-intensive initializations at once).
    ///
    /// 3. **Failure handling (fail-fast and automatic pruning)**:
    ///    - If any shard's `setup()` fails:
    ///      - That shard is marked FAILED.
    ///      - All downstream shards (direct and transitive dependents) are automatically **skipped**
    ///        and never started (via `notify_failure` + `SKIPPED` propagation).
    ///    - This prevents cascading failures and resource waste on shards that can no longer function.
    ///    - Failed and skipped shards are recorded in the returned `StartupReport`.
    ///
    /// 4. **Event-driven execution loop**:
    ///    - Uses `FuturesUnordered` + a manual `ready_queue` (VecDeque) to:
    ///      - Fill up to `max_concurrency` tasks when possible
    ///      - Process completions asynchronously
    ///      - Immediately enqueue newly ready shards upon success
    ///    - The loop continues until no more tasks are pending and no ready shards remain.
    ///
    /// 5. **Thread-safety and lifetime**:
    ///    - `Arc<Jax>` and `Arc<ShardScheduler>` are cloned into each async task.
    ///
    /// ### Returned value
    /// - `Ok(StartupReport)` on completion (even with failures/skips)
    /// - Contains lists of:
    ///   - `failed`: shards that errored during setup
    ///   - `skipped`: shards that were never started due to upstream failure
    ///
    /// ### When to call
    /// Typically invoked once during application startup (e.g., in Tauri's `setup()` hook)
    /// after all shards have been `register()`ed and `build()` has been called.
    pub async fn start(self: &Arc<Self>) -> Result<StartupReport, Box<dyn Error + Send + Sync>> {
        let snapshot = self.snapshot_registry().ok_or("Jax: Registry null")?;

        // 1. Wrap the scheduler in an Arc to share it across multiple Futures
        let scheduler = Arc::new(ShardScheduler::new(&snapshot.graph));
        let mut tasks = FuturesUnordered::new();
        let mut report = StartupReport::default();

        let mut ready_queue = VecDeque::new();
        let max_concurrency = self.config.max_concurrency;

        // 2. Use normal reference capture for the closure, or clone explicitly
        // Note: 'move' is removed before the closure, and cloning is done inside
        let make_task = |shard: Arc<dyn Shard>, idx: usize| {
            let jax_ptr = Arc::clone(self);
            let scheduler_ptr = Arc::clone(&scheduler);
            let shard_id = shard.id();
            async move {
                scheduler_ptr.mark_running(idx);
                match shard.setup(jax_ptr).await {
                    Ok(_) => Ok((shard_id, idx)),
                    Err(e) => Err((shard_id, idx, e)),
                }
            }
        };

        for item in scheduler.collect_seeds() {
            ready_queue.push_back(item);
        }

        // 4. Event-driven loop
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

        Ok(report)
    }

    /// Graceful shutdown of all shards using **best-effort reverse topological order**.
    ///
    /// ### Strategy (why this design):
    /// 1. **Reverse topological order** (layers are computed with Kahn's algorithm then reversed):
    ///    - Consumer shards (leaves) teardown **first**.
    ///    - Provider shards (roots) teardown **last**.
    ///      This guarantees that no shard is still using a resource that has already been cleaned up.
    ///
    /// 2. **Best-effort / tolerant policy** (the most important part):
    ///    - Every single `teardown()` in every layer **will run to completion**.
    ///    - Failure in one shard **never cancels** other shards in the same layer or later layers.
    ///    - This prevents resource leaks (sockets, event subscriptions, threads, files, etc.)
    ///      even when some shards fail to shut down cleanly.
    ///
    /// 3. **Error handling**:
    ///    - All errors are collected internally.
    ///    - Only the **first** error is returned (others are discarded or logged).
    ///
    /// ### Concurrency:
    /// - Inside each layer: `buffer_unordered(max_concurrency)` (safe because same-layer shards have no dependencies).
    ///
    pub async fn stop(self: &Arc<Self>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let snapshot = match self.snapshot_registry() {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut layers = compute_shard_layers(&snapshot.graph)?;
        layers.reverse();

        let max_concurrency = self.config.max_concurrency;
        let mut all_errors: Vec<Box<dyn Error + Send + Sync>> = Vec::new();

        for layer in layers.into_iter() {
            let results = stream::iter(layer)
                .map(|shard| {
                    let jax_ptr = Arc::clone(self);
                    async move { shard.teardown(jax_ptr).await }
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

    /// Retrieve a native shard by its concrete type.
    ///
    /// ```ignore
    /// let lcu = jax.get_shard::<LcuShard>();
    /// lcu.subscribe();
    /// ```
    ///
    /// Panics if `T` was never registered.
    pub fn get_shard<T: Shard>(&self) -> Arc<T> {
        let snapshot = self
            .snapshot_registry()
            .expect("Jax: Registry is null. Did you call build()?");

        let shard_uuid = T::static_id();

        let node_idx = snapshot.native_indices.get(&shard_uuid).unwrap_or_else(|| {
            panic!(
                "Jax: Shard [{}] with ID [{}] not found. Was it registered?",
                type_name::<T>(),
                shard_uuid
            )
        });

        let shard = snapshot.graph[*node_idx].clone();

        shard.downcast_arc::<T>().unwrap_or_else(|_| {
            panic!(
                "Jax: Shard type [{}] mismatch for ID [{}]",
                type_name::<T>(),
                shard_uuid
            )
        })
    }

    fn snapshot_registry(&self) -> Option<Arc<ShardRegistry>> {
        let ptr = self.registry.load(Ordering::Acquire);

        if ptr.is_null() {
            return None;
        }

        unsafe {
            Arc::increment_strong_count(ptr);
            Some(Arc::from_raw(ptr))
        }
    }
}

impl Drop for Jax {
    fn drop(&mut self) {
        let ptr = self.registry.swap(core::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            // SAFETY: ptr was created by Arc::into_raw in build()
            unsafe {
                drop(Arc::from_raw(ptr));
            }
        }
    }
}
