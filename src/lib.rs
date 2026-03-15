#![no_std]
extern crate alloc;

pub mod report;
pub mod shard;

use crate::report::{ShardError, StartupReport};
use crate::shard::Shard;
use crate::shard::dag::compute_shard_layers;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
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
}

impl Jax {
    /// Register a shard. Must be called before `start()`.
    pub fn register<T: Shard>(&mut self, shard: Arc<T>) {
        self.pending.push(shard);
    }

    /// Finalizes the registry by building the dependency graph.
    /// This is an incremental-rebuild: it loads the current registry, clones its data, and merges the new shards.
    pub fn build(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
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

        Ok(())
    }

    pub async fn start(self: &Arc<Self>) -> Result<StartupReport, Box<dyn Error + Send + Sync>> {
        let snapshot = self
            .snapshot_registry()
            .ok_or("Jax: Registry is null. Did you call build()?")?;

        // Compute execution layers via DAG topological sort
        let layers = compute_shard_layers(&snapshot.graph)?;

        // Set up shards layer by layer; within each layer shards are independent
        let mut report = StartupReport::default();
        let mut failed_ids = BTreeSet::new();

        for layer in layers {
            let mut futures = Vec::new();

            for shard in layer {
                let shard_id = shard.id();

                let has_failed_dependency = shard
                    .dependencies()
                    .iter()
                    .any(|dep_id| failed_ids.contains(dep_id));

                if has_failed_dependency {
                    failed_ids.insert(shard_id);
                    report.skipped.push(shard_id);
                    continue;
                }

                let jax_ref = Arc::clone(self);
                futures.push(async move {
                    match shard.setup(jax_ref).await {
                        Ok(_) => Ok(shard_id),
                        Err(e) => Err(ShardError {
                            id: shard_id,
                            error: e,
                        }),
                    }
                });
            }

            // Setup current layer
            let results = futures::future::join_all(futures).await;

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
        let ptr = self.registry.load(Ordering::Acquire);
        if ptr.is_null() {
            return Ok(());
        }

        let snapshot = self
            .snapshot_registry()
            .expect("Jax: Failed to load shard registry");

        // Compute execution layers
        let mut layers = compute_shard_layers(&snapshot.graph)?;

        // Teardown in reverse order (bottom-up)
        layers.reverse();

        for layer in layers {
            let mut futures = Vec::new();
            for shard in layer {
                futures.push(async move { shard.teardown().await });
            }

            let results = futures::future::join_all(futures).await;
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
