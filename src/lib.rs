#![no_std]
extern crate alloc;

pub mod shard;

use crate::shard::Shard;
use crate::shard::dag::compute_shard_layers;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
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
        self.pending.push(shard as Arc<dyn Shard>);
    }

    pub fn build(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut graph = StableDiGraph::new();
        let mut indices = BTreeMap::new();

        // 1. Drain all pending shards
        let shards: Vec<Arc<dyn Shard>> = self.pending.drain(..).collect();

        // 2. Add each shard as a graph node
        for shard in shards {
            let shard_id = shard.id();
            if indices.contains_key(&shard_id) {
                return Err("Duplicate shard registered".into());
            }
            let idx = graph.add_node(shard);
            indices.insert(shard_id, idx);
        }

        // 3. Build dependency edges
        for &consumer_idx in indices.values() {
            let shard = &graph[consumer_idx];
            for dep_id in shard.dependencies() {
                if let Some(&provider_idx) = indices.get(&dep_id) {
                    graph.add_edge(provider_idx, consumer_idx, ());
                } else {
                    return Err("Missing dependency found during build".into());
                }
            }
        }

        // 4. Cycle detection
        if petgraph::algo::is_cyclic_directed(&graph) {
            return Err("Jax: Circular dependency detected!".into());
        }

        // 5. Wrap into a new snapshot
        let new_registry = Arc::new(ShardRegistry {
            graph,
            native_indices: indices,
            guest_indices: Default::default(),
        });
        let new_ptr = Arc::into_raw(new_registry) as *mut ShardRegistry;

        // 6. Atomically swap the registry pointer
        let old_ptr = self.registry.swap(new_ptr, Ordering::AcqRel);

        // 7. Drop the old registry if it exists
        if !old_ptr.is_null() {
            // SAFETY: `old_ptr` was created by `Arc::into_raw` in a previous `build()` call.
            // The `swap` above ensures exclusive ownership of this pointer. Reconstructing
            // the Arc decrements the strong count, allowing deallocation when no longer used.
            unsafe {
                Arc::from_raw(old_ptr);
            }
        }

        Ok(())
    }

    pub async fn start(self: &Arc<Self>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let ptr = self.registry.load(Ordering::Acquire);
        if ptr.is_null() {
            return Err("Jax: Registry is null. Did you call build()?".into());
        }

        // SAFETY: `ptr` was created by `Arc::into_raw` in `build()`.
        // `increment_strong_count` keeps the original alive while `from_raw` produces
        // a second owning Arc for local use.
        let snapshot = unsafe {
            Arc::increment_strong_count(ptr);
            Arc::from_raw(ptr)
        };

        // Compute execution layers via DAG topological sort
        let layers = compute_shard_layers(&snapshot.graph)?;

        // Set up shards layer by layer; within each layer shards are independent
        for layer in layers {
            let mut futures = Vec::new();

            for shard in layer {
                let jax_ref = Arc::clone(self);
                futures.push(async move { shard.setup(jax_ref).await });
            }

            // Execute all setups in the same layer concurrently.
            // If any setup fails, the entire start is aborted.
            let results = futures::future::join_all(futures).await;
            for res in results {
                res?;
            }
        }

        Ok(())
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
    pub fn get_shard<T: Shard + Send + Sync + 'static>(&self) -> Arc<T> {
        let snapshot = self
            .snapshot_registry()
            .expect("Jax: Registry is null. Did you call build()?");

        let shard_uuid = T::static_id();

        let node_idx = snapshot
            .native_indices
            .get(&shard_uuid)
            .unwrap_or_else(|| {
                panic!("Jax: Shard with ID [{}] not found. Was it registered?", shard_uuid)
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
