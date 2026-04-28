use crate::config::JaxConfig;
use crate::probe::Probe;
use crate::registry::{RegistryHandle, ShardRegistry};
use crate::report::StoredStartupReport;
use crate::shard::{Shard, ShardId, TypedShard};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::type_name;
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
use petgraph::prelude::*;
use petgraph::visit::EdgeRef;

/// Core application context.
///
/// Lifecycle: `default/with_config()` -> `probe()` -> `register()` -> `build()` -> `Arc::new()` -> `start()`.
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
    pub fn register(mut self, shard: Arc<dyn Shard>) -> Self {
        self.pending.push(shard);
        self
    }

    /// Finalizes the registry by building the dependency graph.
    pub fn build(mut self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        if self.pending.is_empty() {
            return Ok(self);
        }

        let (mut graph, mut indices) = self.registry_graph_parts();

        let new_shards: Vec<Arc<dyn Shard>> = self.pending.drain(..).collect();
        let mut new_node_indices = Vec::with_capacity(new_shards.len());

        for shard in new_shards {
            let shard_id = shard.descriptor().id();

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
            let dependencies = {
                let shard = &graph[consumer_idx];
                shard.descriptor().dependencies().to_vec()
            };

            for dependency in dependencies {
                let dep_id = dependency.id();
                if let Some(&provider_idx) = indices.get(&dep_id) {
                    graph.add_edge(provider_idx, consumer_idx, ());
                } else {
                    let consumer_id = graph[consumer_idx].descriptor().id();
                    return Err(alloc::format!(
                        "Jax: Missing dependency [{}] for new shard [{}]",
                        dep_id,
                        consumer_id
                    )
                    .into());
                }
            }
        }

        if petgraph::algo::is_cyclic_directed(&graph) {
            return Err("Jax: Incremental build failed due to circular dependency".into());
        }

        self.registry
            .swap(Arc::new(ShardRegistry { graph, indices }));

        Ok(self)
    }

    /// Retrieve a typed shard by its concrete Rust type.
    ///
    /// Panics if `T` was never registered or the registered shard has a
    /// mismatched concrete type for `T::static_id()`.
    pub fn get_shard<T: TypedShard>(&self) -> Arc<T> {
        let snapshot = self
            .registry
            .snapshot()
            .expect("Jax: Registry is null. Did you call build()?");

        let shard_uuid = T::static_id();

        let node_idx = snapshot.indices.get(&shard_uuid).unwrap_or_else(|| {
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
    pub fn list_shards(&self) -> Vec<(ShardId, String, Vec<ShardId>)> {
        let Some(snapshot) = self.registry.snapshot() else {
            return Vec::new();
        };
        snapshot
            .indices
            .keys()
            .map(|&id| {
                let idx = snapshot.indices[&id];
                let descriptor = snapshot.graph[idx].descriptor();
                let label = descriptor.label().into();
                let deps = descriptor
                    .dependencies()
                    .iter()
                    .map(|dependency| dependency.id())
                    .collect();
                (id, label, deps)
            })
            .collect()
    }

    /// Mount a shard at runtime and immediately run its setup lifecycle.
    pub async fn mount(
        self: &Arc<Self>,
        shard: Arc<dyn Shard>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shard_id = shard.descriptor().id();
        let (mut graph, mut indices) = self.registry_graph_parts();

        if indices.contains_key(&shard_id) {
            return Err(
                alloc::format!("Jax: Shard [{}] already exists in registry", shard_id).into(),
            );
        }

        let dependencies = shard.descriptor().dependencies().to_vec();
        let shard_idx = graph.add_node(Arc::clone(&shard));
        indices.insert(shard_id, shard_idx);

        for dependency in dependencies {
            let dep_id = dependency.id();
            if let Some(&provider_idx) = indices.get(&dep_id) {
                graph.add_edge(provider_idx, shard_idx, ());
            } else {
                return Err(alloc::format!(
                    "Jax: Missing dependency [{}] for mounted shard [{}]",
                    dep_id,
                    shard_id
                )
                .into());
            }
        }

        if petgraph::algo::is_cyclic_directed(&graph) {
            return Err("Jax: Mount failed due to circular dependency".into());
        }

        for probe in &self.config.probes {
            probe.before_setup(shard.as_ref()).await?;
        }

        let setup_result = shard.setup(Arc::clone(self)).await;

        for probe in &self.config.probes {
            probe.after_setup(shard.as_ref(), &setup_result).await;
        }

        setup_result?;

        self.registry
            .swap(Arc::new(ShardRegistry { graph, indices }));

        Ok(())
    }

    /// Unmount a shard at runtime after running its teardown lifecycle.
    ///
    /// This conservative operation refuses to unmount a shard that still has
    /// dependents in the registry.
    pub async fn unmount(
        self: &Arc<Self>,
        shard_id: ShardId,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let snapshot = self.registry.snapshot().ok_or("Jax: Registry null")?;
        let mut graph = snapshot.graph.clone();
        let mut indices = snapshot.indices.clone();
        let shard_idx = *indices
            .get(&shard_id)
            .ok_or_else(|| alloc::format!("Jax: Shard [{}] not found", shard_id))?;

        let dependents: Vec<ShardId> = graph
            .edges_directed(shard_idx, Outgoing)
            .map(|edge| graph[edge.target()].descriptor().id())
            .collect();
        if !dependents.is_empty() {
            return Err(alloc::format!(
                "Jax: Cannot unmount shard [{}] while dependents remain: {:?}",
                shard_id,
                dependents
            )
            .into());
        }

        let shard = graph[shard_idx].clone();

        for probe in &self.config.probes {
            probe.before_teardown(shard.as_ref()).await;
        }

        let teardown_result = shard.teardown(Arc::clone(self)).await;

        for probe in &self.config.probes {
            probe.after_teardown(shard.as_ref(), &teardown_result).await;
        }

        teardown_result?;

        graph.remove_node(shard_idx);
        indices.remove(&shard_id);

        self.registry
            .swap(Arc::new(ShardRegistry { graph, indices }));

        Ok(())
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

impl Jax {
    fn registry_graph_parts(
        &self,
    ) -> (
        StableDiGraph<Arc<dyn Shard>, ()>,
        BTreeMap<ShardId, NodeIndex>,
    ) {
        if let Some(snapshot) = self.registry.snapshot() {
            (snapshot.graph.clone(), snapshot.indices.clone())
        } else {
            (StableDiGraph::new(), BTreeMap::new())
        }
    }
}
