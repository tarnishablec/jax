use crate::config::JaxConfig;
use crate::probe::Probe;
use crate::registry::{RegistryHandle, ShardLifecycleState, ShardRegistry};
use crate::shard::{Shard, ShardId, TypedShard};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::type_name;
use core::error::Error;
use petgraph::prelude::*;
use petgraph::visit::EdgeRef;
use spin::Mutex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JaxLifecycleState {
    Built,
    Starting,
    Started,
    StartFailed,
    Stopping,
    Stopped,
}

/// Core application context.
///
/// Lifecycle: `default/with_config()` -> `probe()` -> `register()` -> `build()` -> `Arc::new()` -> `start()`.
pub struct Jax {
    pub(crate) registry: RegistryHandle,
    lifecycle: Mutex<JaxLifecycleState>,
    pending: Vec<Arc<dyn Shard>>,
    pub(crate) config: JaxConfig,
}

impl Default for Jax {
    fn default() -> Self {
        Self {
            registry: RegistryHandle::default(),
            lifecycle: Mutex::new(JaxLifecycleState::Built),
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
            lifecycle: Mutex::new(JaxLifecycleState::Built),
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
            if self.registry.snapshot().is_none() {
                self.registry.swap(Arc::new(ShardRegistry::empty()));
            }
            return Ok(self);
        }

        let (mut graph, mut indices, mut states) = self.registry_parts();

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
            states.insert(shard_id, ShardLifecycleState::Registered);
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
            .swap(Arc::new(ShardRegistry::new(graph, indices, states)));

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
        self.reserve_mount(Arc::clone(&shard)).await?;

        for probe in &self.config.probes {
            if let Err(error) = probe.before_setup(shard.as_ref()).await {
                self.rollback_mount(shard_id).await;
                return Err(error);
            }
        }

        let setup_result = shard.setup(Arc::clone(self)).await;

        for probe in &self.config.probes {
            probe.after_setup(shard.as_ref(), &setup_result).await;
        }

        if let Err(error) = setup_result {
            self.rollback_mount(shard_id).await;
            return Err(error);
        }

        self.finish_mount(shard_id).await?;

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
        let shard = self.reserve_unmount(shard_id).await?;

        for probe in &self.config.probes {
            probe.before_teardown(shard.as_ref()).await;
        }

        let teardown_result = shard.teardown(Arc::clone(self)).await;

        for probe in &self.config.probes {
            probe.after_teardown(shard.as_ref(), &teardown_result).await;
        }

        if teardown_result.is_err() {
            self.restore_unmount(shard_id).await;
        }
        teardown_result?;

        self.finish_unmount(shard_id).await?;

        Ok(())
    }
}

impl Jax {
    async fn reserve_mount(
        &self,
        shard: Arc<dyn Shard>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _mutation = self.registry.lock_mutation().await;
        self.ensure_started("mount")?;

        let shard_id = shard.descriptor().id();
        let (mut graph, mut indices, mut states) = self.registry_parts();

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
                if states.get(&dep_id) != Some(&ShardLifecycleState::Started) {
                    return Err(alloc::format!(
                        "Jax: Dependency [{}] for mounted shard [{}] is not started",
                        dep_id,
                        shard_id
                    )
                    .into());
                }
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

        states.insert(shard_id, ShardLifecycleState::Starting);
        self.registry
            .swap(Arc::new(ShardRegistry::new(graph, indices, states)));

        Ok(())
    }

    async fn finish_mount(&self, shard_id: ShardId) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _mutation = self.registry.lock_mutation().await;
        let snapshot = self.registry.snapshot().ok_or("Jax: Registry null")?;

        if !snapshot.indices.contains_key(&shard_id) {
            return Err(alloc::format!(
                "Jax: Mounted shard [{}] disappeared before setup completed",
                shard_id
            )
            .into());
        }

        snapshot.set_state(shard_id, ShardLifecycleState::Started);
        Ok(())
    }

    async fn rollback_mount(&self, shard_id: ShardId) {
        let _mutation = self.registry.lock_mutation().await;
        let Some(snapshot) = self.registry.snapshot() else {
            return;
        };

        if snapshot.state(shard_id) != Some(ShardLifecycleState::Starting) {
            return;
        }

        let mut graph = snapshot.graph.clone();
        let mut indices = snapshot.indices.clone();
        let mut states = snapshot.states_snapshot();

        if let Some(shard_idx) = indices.remove(&shard_id) {
            graph.remove_node(shard_idx);
            states.remove(&shard_id);
            self.registry
                .swap(Arc::new(ShardRegistry::new(graph, indices, states)));
        }
    }

    async fn reserve_unmount(
        &self,
        shard_id: ShardId,
    ) -> Result<Arc<dyn Shard>, Box<dyn Error + Send + Sync>> {
        let _mutation = self.registry.lock_mutation().await;
        self.ensure_started("unmount")?;

        let snapshot = self.registry.snapshot().ok_or("Jax: Registry null")?;
        let shard_idx = *snapshot
            .indices
            .get(&shard_id)
            .ok_or_else(|| alloc::format!("Jax: Shard [{}] not found", shard_id))?;

        if snapshot.state(shard_id) != Some(ShardLifecycleState::Started) {
            return Err(alloc::format!(
                "Jax: Cannot unmount shard [{}] while it is {:?}",
                shard_id,
                snapshot.state(shard_id)
            )
            .into());
        }

        let dependents: Vec<ShardId> = snapshot
            .graph
            .edges_directed(shard_idx, Outgoing)
            .map(|edge| snapshot.graph[edge.target()].descriptor().id())
            .collect();
        if !dependents.is_empty() {
            return Err(alloc::format!(
                "Jax: Cannot unmount shard [{}] while dependents remain: {:?}",
                shard_id,
                dependents
            )
            .into());
        }

        let shard = snapshot.graph[shard_idx].clone();
        snapshot.set_state(shard_id, ShardLifecycleState::Stopping);
        Ok(shard)
    }

    async fn finish_unmount(&self, shard_id: ShardId) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _mutation = self.registry.lock_mutation().await;
        let snapshot = self.registry.snapshot().ok_or("Jax: Registry null")?;

        if snapshot.state(shard_id) != Some(ShardLifecycleState::Stopping) {
            return Err(alloc::format!(
                "Jax: Cannot finish unmount for shard [{}] while it is {:?}",
                shard_id,
                snapshot.state(shard_id)
            )
            .into());
        }

        let mut graph = snapshot.graph.clone();
        let mut indices = snapshot.indices.clone();
        let mut states = snapshot.states_snapshot();
        let shard_idx = indices
            .remove(&shard_id)
            .ok_or_else(|| alloc::format!("Jax: Shard [{}] not found", shard_id))?;

        graph.remove_node(shard_idx);
        states.remove(&shard_id);

        self.registry
            .swap(Arc::new(ShardRegistry::new(graph, indices, states)));

        Ok(())
    }

    async fn restore_unmount(&self, shard_id: ShardId) {
        let _mutation = self.registry.lock_mutation().await;
        if let Some(snapshot) = self.registry.snapshot() {
            if snapshot.state(shard_id) == Some(ShardLifecycleState::Stopping) {
                snapshot.set_state(shard_id, ShardLifecycleState::Started);
            }
        }
    }

    pub(crate) fn begin_start(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut state = self.lifecycle.lock();
        match *state {
            JaxLifecycleState::Built | JaxLifecycleState::Stopped => {
                *state = JaxLifecycleState::Starting;
                Ok(())
            }
            JaxLifecycleState::Starting => Err("Jax: runtime is already starting".into()),
            JaxLifecycleState::Started => Err("Jax: runtime is already started".into()),
            JaxLifecycleState::StartFailed => {
                Err("Jax: runtime has started shards from a failed start; call stop() before restarting".into())
            }
            JaxLifecycleState::Stopping => Err("Jax: runtime is stopping".into()),
        }
    }

    pub(crate) fn finish_start(&self, started: bool) {
        let mut state = self.lifecycle.lock();
        *state = if started {
            JaxLifecycleState::Started
        } else {
            JaxLifecycleState::StartFailed
        };
    }

    pub(crate) fn begin_stop(&self) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let mut state = self.lifecycle.lock();
        match *state {
            JaxLifecycleState::Started | JaxLifecycleState::StartFailed => {
                *state = JaxLifecycleState::Stopping;
                Ok(true)
            }
            JaxLifecycleState::Built | JaxLifecycleState::Stopped => Ok(false),
            JaxLifecycleState::Starting => Err("Jax: runtime is starting".into()),
            JaxLifecycleState::Stopping => Err("Jax: runtime is already stopping".into()),
        }
    }

    pub(crate) fn finish_stop(&self) {
        *self.lifecycle.lock() = JaxLifecycleState::Stopped;
    }

    pub(crate) fn abort_stop(&self) {
        *self.lifecycle.lock() = JaxLifecycleState::Started;
    }

    fn ensure_started(&self, operation: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let state = *self.lifecycle.lock();
        if state == JaxLifecycleState::Started {
            Ok(())
        } else {
            Err(alloc::format!("Jax: cannot {operation} while runtime is {:?}", state).into())
        }
    }

    fn registry_parts(
        &self,
    ) -> (
        StableDiGraph<Arc<dyn Shard>, ()>,
        BTreeMap<ShardId, NodeIndex>,
        BTreeMap<ShardId, ShardLifecycleState>,
    ) {
        if let Some(snapshot) = self.registry.snapshot() {
            (
                snapshot.graph.clone(),
                snapshot.indices.clone(),
                snapshot.states_snapshot(),
            )
        } else {
            (StableDiGraph::new(), BTreeMap::new(), BTreeMap::new())
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::{Descriptor, JaxResult, TypedShard, depends, shard_id};
    use alloc::string::ToString;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    static REMOVED_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static REMAINING_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static REMAINING_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static MOUNT_A_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static MOUNT_B_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static STOP_ONCE_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static PARTIAL_STARTED_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static PARTIAL_STARTED_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static SKIPPED_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static SKIPPED_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static ZERO_CONCURRENCY_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static ZERO_CONCURRENCY_TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static SELF_VISIBLE_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static REENTRANT_OUTER_SETUPS: AtomicUsize = AtomicUsize::new(0);
    static REENTRANT_INNER_SETUPS: AtomicUsize = AtomicUsize::new(0);

    macro_rules! counted_yield_setup_shard {
        ($ty:ident, $uuid:literal, $counter:ident) => {
            struct $ty;

            impl TypedShard for $ty {
                shard_id!($uuid);
            }

            #[async_trait::async_trait]
            impl Shard for $ty {
                fn descriptor(&self) -> Descriptor {
                    Descriptor::typed::<Self>()
                }

                async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
                    $counter.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                    Ok(())
                }
            }
        };
    }

    struct RemovedShard;

    impl TypedShard for RemovedShard {
        shard_id!("00000000-0000-0000-0000-000000000101");
    }

    #[async_trait::async_trait]
    impl Shard for RemovedShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REMOVED_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct RemainingShard;

    impl TypedShard for RemainingShard {
        shard_id!("00000000-0000-0000-0000-000000000102");
    }

    #[async_trait::async_trait]
    impl Shard for RemainingShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REMAINING_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REMAINING_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    counted_yield_setup_shard!(
        MountA,
        "00000000-0000-0000-0000-000000000201",
        MOUNT_A_SETUPS
    );
    counted_yield_setup_shard!(
        MountB,
        "00000000-0000-0000-0000-000000000202",
        MOUNT_B_SETUPS
    );

    struct FailStartShard;

    impl TypedShard for FailStartShard {
        shard_id!("00000000-0000-0000-0000-000000000301");
    }

    #[async_trait::async_trait]
    impl Shard for FailStartShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            Err("intentional setup failure".into())
        }
    }

    struct StopOnceShard;

    impl TypedShard for StopOnceShard {
        shard_id!("00000000-0000-0000-0000-000000000302");
    }

    #[async_trait::async_trait]
    impl Shard for StopOnceShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            STOP_ONCE_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct PartialStartedShard;

    impl TypedShard for PartialStartedShard {
        shard_id!("00000000-0000-0000-0000-000000000303");
    }

    #[async_trait::async_trait]
    impl Shard for PartialStartedShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            PARTIAL_STARTED_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            PARTIAL_STARTED_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct SkippedAfterFailShard;

    impl TypedShard for SkippedAfterFailShard {
        shard_id!("00000000-0000-0000-0000-000000000304");
    }

    #[async_trait::async_trait]
    impl Shard for SkippedAfterFailShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>().with_dependencies(depends![FailStartShard])
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            SKIPPED_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            SKIPPED_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct ZeroConcurrencyShard;

    impl TypedShard for ZeroConcurrencyShard {
        shard_id!("00000000-0000-0000-0000-000000000305");
    }

    #[async_trait::async_trait]
    impl Shard for ZeroConcurrencyShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            ZERO_CONCURRENCY_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            ZERO_CONCURRENCY_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct SelfVisibleMountShard;

    impl TypedShard for SelfVisibleMountShard {
        shard_id!("00000000-0000-0000-0000-000000000306");
    }

    #[async_trait::async_trait]
    impl Shard for SelfVisibleMountShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, jax: Arc<Jax>) -> JaxResult<()> {
            let _self_shard = jax.get_shard::<SelfVisibleMountShard>();
            SELF_VISIBLE_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct ReentrantOuterShard;

    impl TypedShard for ReentrantOuterShard {
        shard_id!("00000000-0000-0000-0000-000000000307");
    }

    #[async_trait::async_trait]
    impl Shard for ReentrantOuterShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, jax: Arc<Jax>) -> JaxResult<()> {
            REENTRANT_OUTER_SETUPS.fetch_add(1, Ordering::Relaxed);
            jax.mount(Arc::new(ReentrantInnerShard)).await
        }
    }

    struct ReentrantInnerShard;

    impl TypedShard for ReentrantInnerShard {
        shard_id!("00000000-0000-0000-0000-000000000308");
    }

    #[async_trait::async_trait]
    impl Shard for ReentrantInnerShard {
        fn descriptor(&self) -> Descriptor {
            Descriptor::typed::<Self>()
        }

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REENTRANT_INNER_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn lifecycle_runs_after_unmount_leaves_sparse_node_indices() -> JaxResult<()> {
        REMOVED_TEARDOWNS.store(0, Ordering::Relaxed);
        REMAINING_SETUPS.store(0, Ordering::Relaxed);
        REMAINING_TEARDOWNS.store(0, Ordering::Relaxed);

        let jax = Jax::default()
            .register(Arc::new(RemovedShard))
            .register(Arc::new(RemainingShard))
            .build()?;
        let jax = Arc::new(jax);

        let report = jax.start().await?;
        assert!(report.is_success());
        jax.unmount(RemovedShard::static_id()).await?;
        jax.stop().await?;

        let report = jax.start().await?;
        assert!(report.is_success());
        jax.stop().await?;

        assert_eq!(REMOVED_TEARDOWNS.load(Ordering::Relaxed), 1);
        assert_eq!(REMAINING_SETUPS.load(Ordering::Relaxed), 2);
        assert_eq!(REMAINING_TEARDOWNS.load(Ordering::Relaxed), 2);

        Ok(())
    }

    #[tokio::test]
    async fn concurrent_mounts_are_serialized_without_lost_updates() -> JaxResult<()> {
        MOUNT_A_SETUPS.store(0, Ordering::Relaxed);
        MOUNT_B_SETUPS.store(0, Ordering::Relaxed);

        let jax = Arc::new(Jax::default().build()?);
        let report = jax.start().await?;
        assert!(report.is_success());

        let mount_a = jax.mount(Arc::new(MountA));
        let mount_b = jax.mount(Arc::new(MountB));

        let (result_a, result_b) = tokio::join!(mount_a, mount_b);
        result_a?;
        result_b?;

        let shard_ids = jax
            .list_shards()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>();

        assert!(shard_ids.contains(&MountA::static_id()));
        assert!(shard_ids.contains(&MountB::static_id()));
        assert_eq!(MOUNT_A_SETUPS.load(Ordering::Relaxed), 1);
        assert_eq!(MOUNT_B_SETUPS.load(Ordering::Relaxed), 1);

        Ok(())
    }

    #[tokio::test]
    async fn mount_requires_started_runtime() -> JaxResult<()> {
        let jax = Arc::new(Jax::default().build()?);

        let error = jax
            .mount(Arc::new(MountA))
            .await
            .expect_err("mount before start should fail");

        assert!(error.to_string().contains("cannot mount"));
        Ok(())
    }

    #[tokio::test]
    async fn repeated_start_is_rejected_until_stop() -> JaxResult<()> {
        let jax = Arc::new(Jax::default().build()?);

        let report = jax.start().await?;
        assert!(report.is_success());

        let error = match jax.start().await {
            Ok(_) => panic!("second start should fail while started"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already started"));

        jax.stop().await?;
        let report = jax.start().await?;
        assert!(report.is_success());

        Ok(())
    }

    #[tokio::test]
    async fn repeated_stop_is_idempotent() -> JaxResult<()> {
        STOP_ONCE_TEARDOWNS.store(0, Ordering::Relaxed);

        let jax = Jax::default().register(Arc::new(StopOnceShard)).build()?;
        let jax = Arc::new(jax);

        let report = jax.start().await?;
        assert!(report.is_success());

        jax.stop().await?;
        jax.stop().await?;

        assert_eq!(STOP_ONCE_TEARDOWNS.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[tokio::test]
    async fn failed_start_does_not_enter_started_state() -> JaxResult<()> {
        let jax = Jax::default().register(Arc::new(FailStartShard)).build()?;
        let jax = Arc::new(jax);

        let report = jax.start().await?;
        assert!(!report.is_success());

        let error = jax
            .mount(Arc::new(MountA))
            .await
            .expect_err("mount after failed start should fail");
        assert!(error.to_string().contains("cannot mount"));

        Ok(())
    }

    #[tokio::test]
    async fn stop_after_failed_start_tears_down_only_started_shards() -> JaxResult<()> {
        PARTIAL_STARTED_SETUPS.store(0, Ordering::Relaxed);
        PARTIAL_STARTED_TEARDOWNS.store(0, Ordering::Relaxed);
        SKIPPED_SETUPS.store(0, Ordering::Relaxed);
        SKIPPED_TEARDOWNS.store(0, Ordering::Relaxed);

        let jax = Jax::default()
            .register(Arc::new(PartialStartedShard))
            .register(Arc::new(FailStartShard))
            .register(Arc::new(SkippedAfterFailShard))
            .build()?;
        let jax = Arc::new(jax);

        let report = jax.start().await?;
        assert!(!report.is_success());
        assert!(
            report
                .failed
                .iter()
                .any(|failed| failed.id == FailStartShard::static_id())
        );
        assert!(report.skipped.contains(&SkippedAfterFailShard::static_id()));

        jax.stop().await?;

        assert_eq!(PARTIAL_STARTED_SETUPS.load(Ordering::Relaxed), 1);
        assert_eq!(PARTIAL_STARTED_TEARDOWNS.load(Ordering::Relaxed), 1);
        assert_eq!(SKIPPED_SETUPS.load(Ordering::Relaxed), 0);
        assert_eq!(SKIPPED_TEARDOWNS.load(Ordering::Relaxed), 0);

        Ok(())
    }

    #[tokio::test]
    async fn zero_max_concurrency_still_runs_lifecycle() -> JaxResult<()> {
        ZERO_CONCURRENCY_SETUPS.store(0, Ordering::Relaxed);
        ZERO_CONCURRENCY_TEARDOWNS.store(0, Ordering::Relaxed);

        let mut config = JaxConfig::default();
        config.max_concurrency = 0;
        let jax = Jax::with_config(config)
            .register(Arc::new(ZeroConcurrencyShard))
            .build()?;
        let jax = Arc::new(jax);

        let report = jax.start().await?;
        assert!(report.is_success());
        jax.stop().await?;

        assert_eq!(ZERO_CONCURRENCY_SETUPS.load(Ordering::Relaxed), 1);
        assert_eq!(ZERO_CONCURRENCY_TEARDOWNS.load(Ordering::Relaxed), 1);

        Ok(())
    }

    #[tokio::test]
    async fn mounted_shard_is_visible_during_setup() -> JaxResult<()> {
        SELF_VISIBLE_SETUPS.store(0, Ordering::Relaxed);

        let jax = Arc::new(Jax::default().build()?);
        let report = jax.start().await?;
        assert!(report.is_success());

        jax.mount(Arc::new(SelfVisibleMountShard)).await?;

        assert_eq!(SELF_VISIBLE_SETUPS.load(Ordering::Relaxed), 1);
        assert!(
            jax.list_shards()
                .into_iter()
                .any(|(id, _, _)| id == SelfVisibleMountShard::static_id())
        );

        Ok(())
    }

    #[tokio::test]
    async fn mount_setup_can_reenter_mount_without_self_deadlock() -> JaxResult<()> {
        REENTRANT_OUTER_SETUPS.store(0, Ordering::Relaxed);
        REENTRANT_INNER_SETUPS.store(0, Ordering::Relaxed);

        let jax = Arc::new(Jax::default().build()?);
        let report = jax.start().await?;
        assert!(report.is_success());

        jax.mount(Arc::new(ReentrantOuterShard)).await?;
        let shard_ids = jax
            .list_shards()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>();

        assert!(shard_ids.contains(&ReentrantOuterShard::static_id()));
        assert!(shard_ids.contains(&ReentrantInnerShard::static_id()));
        assert_eq!(REENTRANT_OUTER_SETUPS.load(Ordering::Relaxed), 1);
        assert_eq!(REENTRANT_INNER_SETUPS.load(Ordering::Relaxed), 1);

        Ok(())
    }
}
