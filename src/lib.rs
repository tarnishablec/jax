#![no_std]
extern crate alloc;

pub mod config;
pub mod report;
pub mod shard;

use crate::config::JaxConfig;
use crate::report::{ShardError, StartupReport};
use crate::shard::Shard;
use crate::shard::dag::compute_shard_layers;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
use futures::stream;
use futures::stream::StreamExt;
use petgraph::stable_graph::StableDiGraph;
use uuid::Uuid;

struct ShardRegistry {
    pub(crate) graph: StableDiGraph<Arc<dyn Shard>, ()>,
    pub(crate) native_indices: BTreeMap<Uuid, petgraph::graph::NodeIndex>,
    #[allow(dead_code)]
    pub(crate) guest_indices: BTreeMap<Uuid, petgraph::graph::NodeIndex>,
}

/// Core application context.
///
/// Lifecycle: `new()` → `register()` → `Arc::new()` → `start()`.
/// Tauri drives it: `Builder::setup()` builds and starts Jax.
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

    /// Starts all registered shards in topological order.
    ///
    /// Execution is performed layer-by-layer. Shards within the same layer are
    /// independent and executed concurrently with a maximum concurrency limit.
    pub async fn start(self: &Arc<Self>) -> Result<StartupReport, Box<dyn Error + Send + Sync>> {
        let snapshot = self
            .snapshot_registry()
            .ok_or("Jax: Registry is null. Did you call build()?")?;

        // 1. Compute execution layers via DAG topological sort.
        // Shards in the same layer have no dependencies on each other.
        let layers = compute_shard_layers(&snapshot.graph)?;

        let mut report = StartupReport::default();
        let mut failed_ids = BTreeSet::new();

        // Define the maximum number of concurrent setups within a single layer.
        // This prevents resource exhaustion (e.g., too many open file handles).
        let max_concurrency = self.config.max_concurrency;

        for layer in layers {
            let mut tasks = Vec::new();

            // 2. Synchronous Phase: Filter shards based on dependency health.
            for shard in layer {
                let shard_id = shard.id();

                let has_failed_dependency = shard
                    .dependencies()
                    .iter()
                    .any(|dep_id| failed_ids.contains(dep_id));

                if has_failed_dependency {
                    // Mark as skipped and propagate failure to its downstream dependents.
                    failed_ids.insert(shard_id);
                    report.skipped.push(shard_id);
                } else {
                    // Shard is healthy; prepare its setup task.
                    let jax_ref = Arc::clone(self);
                    tasks.push(async move {
                        shard
                            .setup(jax_ref)
                            .await
                            .map(|_| shard_id)
                            .map_err(|e| ShardError {
                                id: shard_id,
                                error: e,
                            })
                    });
                }
            }

            // 3. Asynchronous Phase: Execute setups with controlled concurrency.
            // buffer_unordered allows up to `max_concurrency` tasks to run
            // simultaneously without enforcing the completion order.
            let results = stream::iter(tasks)
                .buffer_unordered(max_concurrency)
                .collect::<Vec<_>>()
                .await;

            // 4. Finalize: Collect results and update the failure set.
            for res in results {
                if let Err(err) = res {
                    failed_ids.insert(err.id);
                    report.failed.push(err);
                }
            }
        }

        Ok(report)
    }

    /// Stops all shards in reverse topological order.
    pub async fn stop(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let snapshot = match self.snapshot_registry() {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut layers = compute_shard_layers(&snapshot.graph)?;
        layers.reverse();

        let max_concurrency = self.config.max_concurrency;

        for layer in layers {
            let results = stream::iter(layer)
                .map(|shard| async move { shard.teardown().await })
                .buffer_unordered(max_concurrency)
                .collect::<Vec<_>>()
                .await;

            for res in results {
                res?;
            }
        }

        Ok(())
    }

    /// Retrieve a shard by its concrete type.
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
                "Jax: Shard with ID [{}] not found. Was it registered?",
                shard_uuid
            )
        });

        let shard = snapshot.graph[*node_idx].clone();

        shard
            .downcast_arc::<T>()
            .unwrap_or_else(|_| panic!("Jax: Shard type mismatch for ID [{}]", shard_uuid))
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
