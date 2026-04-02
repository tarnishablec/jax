use crate::config::JaxConfig;
use crate::probe::Probe;
use crate::registry::{RegistryHandle, ShardRegistry};
use crate::report::StoredStartupReport;
use crate::shard::Shard;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::type_name;
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
use petgraph::prelude::*;
use uuid::Uuid;

/// Core application context.
///
/// Lifecycle: `default/with_config()` → `probe()` → `register()` → `build()` → `Arc::new()` → `start()`.
pub struct Jax {
    pub(crate) registry: RegistryHandle,
    pub(crate) startup_report: AtomicPtr<StoredStartupReport>,
    pending: Vec<Arc<dyn Shard>>,
    pub(crate) config: JaxConfig,
}

impl Default for Jax {
    fn default() -> Self {
        Self {
            registry: RegistryHandle::default(),
            startup_report: AtomicPtr::new(core::ptr::null_mut()),
            pending: Vec::new(),
            config: JaxConfig::default(),
        }
    }
}

impl Jax {
    pub fn with_config(config: JaxConfig) -> Self {
        Self {
            config,
            registry: RegistryHandle::default(),
            startup_report: AtomicPtr::new(core::ptr::null_mut()),
            pending: Vec::new(),
        }
    }

    /// Attach a lifecycle probe. Called in registration order.
    ///
    /// The caller retains their `Arc` handle to query probe-specific data
    /// (e.g. `TimingProbe::durations()`) after startup.
    pub fn probe<T: Probe>(mut self, probe: Arc<T>) -> Self {
        self.config.probes.push(probe);
        self
    }

    /// Register a shard. Must be called before `start()`.
    pub fn register<T: Shard>(mut self, shard: Arc<T>) -> Self {
        self.pending.push(shard);
        self
    }

    /// Finalizes the registry by building the dependency graph.
    pub fn build(mut self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        if self.pending.is_empty() {
            return Ok(self);
        }

        let (mut graph, mut indices) = if let Some(snapshot) = self.registry.snapshot() {
            (snapshot.graph.clone(), snapshot.native_indices.clone())
        } else {
            (StableDiGraph::new(), BTreeMap::new())
        };

        let new_shards: Vec<Arc<dyn Shard>> = self.pending.drain(..).collect();
        let mut new_node_indices = Vec::with_capacity(new_shards.len());

        for shard in new_shards {
            let shard_id = shard.id();

            if indices.contains_key(&shard_id) {
                return Err(
                    alloc::format!("Jax: Shard [{}] already exists in registry", shard_id).into(),
                );
            }

            let idx = graph.add_node(shard);
            indices.insert(shard_id, idx);
            new_node_indices.push(idx);
        }

        for &consumer_idx in &new_node_indices {
            let dep_ids = {
                let shard = &graph[consumer_idx];
                shard.dependencies()
            };

            for dep_id in dep_ids {
                if let Some(&provider_idx) = indices.get(&dep_id) {
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

        if petgraph::algo::is_cyclic_directed(&graph) {
            return Err("Jax: Incremental build failed due to circular dependency".into());
        }

        self.registry.swap(Arc::new(ShardRegistry {
            graph,
            native_indices: indices,
            guest_indices: Default::default(),
        }));

        Ok(self)
    }

    /// Retrieve a native shard by its concrete type.
    ///
    /// Panics if `T` was never registered.
    pub fn get_shard<T: Shard>(&self) -> Arc<T> {
        let snapshot = self
            .registry
            .snapshot()
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

    /// Returns all registered shard IDs, labels, and dependency lists.
    pub fn list_shards(&self) -> Vec<(Uuid, String, Vec<Uuid>)> {
        let Some(snapshot) = self.registry.snapshot() else {
            return Vec::new();
        };
        snapshot
            .native_indices
            .keys()
            .map(|&id| {
                let idx = snapshot.native_indices[&id];
                let shard = &snapshot.graph[idx];
                let label = shard.label();
                let deps = shard.dependencies();
                (id, label, deps)
            })
            .collect()
    }

    /// Returns the stored startup report summary (available after `start()` completes).
    pub fn get_startup_report(&self) -> Option<&StoredStartupReport> {
        let ptr = self.startup_report.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(&*ptr) }
        }
    }
}

impl Drop for Jax {
    fn drop(&mut self) {
        let report_ptr = self
            .startup_report
            .swap(core::ptr::null_mut(), Ordering::AcqRel);
        if !report_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(report_ptr));
            }
        }
    }
}
