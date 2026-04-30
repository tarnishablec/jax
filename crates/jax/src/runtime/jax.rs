use crate::config::JaxConfig;
use crate::probe::Probe;
use crate::registry::{RegistryHandle, ShardLifecycleState, ShardRegistry};
use crate::shard::{Shard, ShardId};
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
    Configuring,
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
            lifecycle: Mutex::new(JaxLifecycleState::Configuring),
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
            lifecycle: Mutex::new(JaxLifecycleState::Configuring),
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
        self.begin_build()?;

        if self.pending.is_empty() {
            if self.registry.snapshot().is_none() {
                self.registry.swap(Arc::new(ShardRegistry::empty()));
            }
            self.finish_build();
            return Ok(self);
        }

        let (mut graph, mut indices, mut states) = self.registry_parts();

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
            states.insert(shard_id, ShardLifecycleState::Registered);
            new_node_indices.push(idx);
        }

        for &consumer_idx in &new_node_indices {
            let dependencies = {
                let shard = &graph[consumer_idx];
                shard.dependencies()
            };

            for dep_id in dependencies {
                if let Some(&provider_idx) = indices.get(&dep_id) {
                    graph.add_edge(provider_idx, consumer_idx, ());
                } else {
                    let consumer_id = graph[consumer_idx].id();
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

        self.finish_build();

        Ok(self)
    }

    /// Retrieve a shard by static Rust type ID.
    ///
    /// `T` must declare a static shard ID, usually by implementing its ID with
    /// `shard_id!`. Dynamic-ID shards must be retrieved with
    /// `get_shard_by_id<T>()`.
    ///
    /// Panics if the registry has not been built, if `T` has no static shard
    /// ID, if that ID is not registered, or if the registered shard does not
    /// have type `T`.
    pub fn get_shard<T: Shard>(&self) -> Arc<T> {
        let shard_id = T::static_id().unwrap_or_else(|| {
            panic!(
                "Jax: Shard type [{}] has no static shard ID. Use get_shard_by_id()",
                type_name::<T>()
            )
        });

        let snapshot = match self.registry.snapshot() {
            Some(snapshot) => snapshot,
            None => panic!("Jax: Registry is null. Did you call build()?"),
        };

        Self::get_shard_from_snapshot(&snapshot, shard_id)
    }

    /// Retrieve a shard by stable runtime ID and concrete Rust type.
    ///
    /// Panics if the ID is not registered or if the registered shard has a
    /// different concrete type.
    pub fn get_shard_by_id<T: Shard>(&self, shard_id: ShardId) -> Arc<T> {
        let snapshot = match self.registry.snapshot() {
            Some(snapshot) => snapshot,
            None => panic!("Jax: Registry is null. Did you call build()?"),
        };

        Self::get_shard_from_snapshot(&snapshot, shard_id)
    }

    fn get_shard_from_snapshot<T: Shard>(snapshot: &ShardRegistry, shard_id: ShardId) -> Arc<T> {
        let node_idx = snapshot.indices.get(&shard_id).unwrap_or_else(|| {
            panic!(
                "Jax: Shard [{}] with ID [{}] not found. Was it registered?",
                type_name::<T>(),
                shard_id
            )
        });

        let shard = snapshot.graph[*node_idx].clone();
        shard.downcast_arc::<T>().unwrap_or_else(|_| {
            panic!(
                "Jax: Shard type [{}] mismatch for ID [{}]",
                type_name::<T>(),
                shard_id
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
                let shard = &snapshot.graph[idx];
                let label = shard.label().into();
                let deps = shard.dependencies();
                (id, label, deps)
            })
            .collect()
    }

    /// Mount a shard at runtime and immediately run its setup lifecycle.
    pub async fn mount(
        self: &Arc<Self>,
        shard: Arc<dyn Shard>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shard_id = shard.id();
        self.reserve_mount(Arc::clone(&shard)).await?;

        for probe in &self.config.probes {
            probe.before_setup(shard.as_ref()).await;
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

        let shard_id = shard.id();
        let (mut graph, mut indices, mut states) = self.registry_parts();

        if indices.contains_key(&shard_id) {
            return Err(
                alloc::format!("Jax: Shard [{}] already exists in registry", shard_id).into(),
            );
        }

        let dependencies = shard.dependencies();
        let shard_idx = graph.add_node(Arc::clone(&shard));
        indices.insert(shard_id, shard_idx);

        for dep_id in dependencies {
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
            .map(|edge| snapshot.graph[edge.target()].id())
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
            JaxLifecycleState::Configuring => {
                Err("Jax: runtime is not built. Call build() before start()".into())
            }
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
            JaxLifecycleState::Configuring
            | JaxLifecycleState::Built
            | JaxLifecycleState::Stopped => Ok(false),
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

    fn begin_build(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let state = *self.lifecycle.lock();
        match state {
            JaxLifecycleState::Configuring | JaxLifecycleState::Built => Ok(()),
            JaxLifecycleState::Starting => {
                Err("Jax: cannot build while runtime is starting".into())
            }
            JaxLifecycleState::Started => Err("Jax: cannot build while runtime is started".into()),
            JaxLifecycleState::StartFailed => {
                Err("Jax: cannot build after a failed start; call stop() before rebuilding".into())
            }
            JaxLifecycleState::Stopping => {
                Err("Jax: cannot build while runtime is stopping".into())
            }
            JaxLifecycleState::Stopped => Err("Jax: cannot build after runtime has stopped".into()),
        }
    }

    fn finish_build(&self) {
        *self.lifecycle.lock() = JaxLifecycleState::Built;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::{JaxResult, depends, shard_id};
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
    static DEFAULT_LABEL_SETUPS: AtomicUsize = AtomicUsize::new(0);

    macro_rules! counted_yield_setup_shard {
        ($ty:ident, $uuid:literal, $counter:ident) => {
            struct $ty;

            #[async_trait::async_trait]
            impl Shard for $ty {
                shard_id!($uuid);

                async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
                    $counter.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                    Ok(())
                }
            }
        };
    }

    struct RemovedShard;

    #[async_trait::async_trait]
    impl Shard for RemovedShard {
        shard_id!("00000000-0000-0000-0000-000000000101");

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REMOVED_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct RemainingShard;

    #[async_trait::async_trait]
    impl Shard for RemainingShard {
        shard_id!("00000000-0000-0000-0000-000000000102");

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

    #[async_trait::async_trait]
    impl Shard for FailStartShard {
        shard_id!("00000000-0000-0000-0000-000000000301");

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            Err("intentional setup failure".into())
        }
    }

    struct StopOnceShard;

    #[async_trait::async_trait]
    impl Shard for StopOnceShard {
        shard_id!("00000000-0000-0000-0000-000000000302");

        async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            STOP_ONCE_TEARDOWNS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct PartialStartedShard;

    #[async_trait::async_trait]
    impl Shard for PartialStartedShard {
        shard_id!("00000000-0000-0000-0000-000000000303");

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

    #[async_trait::async_trait]
    impl Shard for SkippedAfterFailShard {
        shard_id!("00000000-0000-0000-0000-000000000304");

        depends![FailStartShard];

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

    #[async_trait::async_trait]
    impl Shard for ZeroConcurrencyShard {
        shard_id!("00000000-0000-0000-0000-000000000305");

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

    #[async_trait::async_trait]
    impl Shard for SelfVisibleMountShard {
        shard_id!("00000000-0000-0000-0000-000000000306");

        async fn setup(&self, jax: Arc<Jax>) -> JaxResult<()> {
            let _self_shard = jax.get_shard::<SelfVisibleMountShard>();
            SELF_VISIBLE_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct ReentrantOuterShard;

    #[async_trait::async_trait]
    impl Shard for ReentrantOuterShard {
        shard_id!("00000000-0000-0000-0000-000000000307");

        async fn setup(&self, jax: Arc<Jax>) -> JaxResult<()> {
            REENTRANT_OUTER_SETUPS.fetch_add(1, Ordering::Relaxed);
            jax.mount(Arc::new(ReentrantInnerShard)).await
        }
    }

    struct ReentrantInnerShard;

    #[async_trait::async_trait]
    impl Shard for ReentrantInnerShard {
        shard_id!("00000000-0000-0000-0000-000000000308");

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            REENTRANT_INNER_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct DynamicIdShard(ShardId);

    #[async_trait::async_trait]
    impl Shard for DynamicIdShard {
        fn id(&self) -> ShardId {
            self.0
        }
    }

    struct DefaultLabelShard;

    #[async_trait::async_trait]
    impl Shard for DefaultLabelShard {
        shard_id!("00000000-0000-0000-0000-000000000405");

        async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
            DEFAULT_LABEL_SETUPS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct MixedDependencyShard;

    #[async_trait::async_trait]
    impl Shard for MixedDependencyShard {
        shard_id!("00000000-0000-0000-0000-000000000406");

        depends![
            FailStartShard,
            "00000000-0000-0000-0000-000000000405".into(),
        ];
    }

    fn test_id(input: &'static str) -> ShardId {
        input
            .parse()
            .unwrap_or_else(|error| panic!("invalid test shard id [{input}]: {error}"))
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
        jax.unmount(test_id("00000000-0000-0000-0000-000000000101"))
            .await?;
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

        assert!(shard_ids.contains(&test_id("00000000-0000-0000-0000-000000000201")));
        assert!(shard_ids.contains(&test_id("00000000-0000-0000-0000-000000000202")));
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
    async fn start_before_build_is_rejected_without_start_failure() -> JaxResult<()> {
        let jax = Arc::new(Jax::default());

        let error = match jax.start().await {
            Ok(_) => panic!("start before build should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("not built"));
        assert_eq!(*jax.lifecycle.lock(), JaxLifecycleState::Configuring);
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
                .any(|failed| failed.id == test_id("00000000-0000-0000-0000-000000000301"))
        );
        assert!(
            report
                .skipped
                .contains(&test_id("00000000-0000-0000-0000-000000000304"))
        );

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
                .any(|(id, _, _)| id == test_id("00000000-0000-0000-0000-000000000306"))
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

        assert!(shard_ids.contains(&test_id("00000000-0000-0000-0000-000000000307")));
        assert!(shard_ids.contains(&test_id("00000000-0000-0000-0000-000000000308")));
        assert_eq!(REENTRANT_OUTER_SETUPS.load(Ordering::Relaxed), 1);
        assert_eq!(REENTRANT_INNER_SETUPS.load(Ordering::Relaxed), 1);

        Ok(())
    }

    #[tokio::test]
    async fn get_shard_uses_static_id() -> JaxResult<()> {
        let shard = Arc::new(DefaultLabelShard);
        let jax = Jax::default().register(shard.clone()).build()?;

        let resolved = jax.get_shard::<DefaultLabelShard>();

        assert!(Arc::ptr_eq(&resolved, &shard));
        Ok(())
    }

    #[tokio::test]
    async fn get_shard_requires_static_id() -> JaxResult<()> {
        let jax = Jax::default()
            .register(Arc::new(DynamicIdShard(test_id(
                "00000000-0000-0000-0000-000000000401",
            ))))
            .build()?;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = jax.get_shard::<DynamicIdShard>();
        }))
        .expect_err("dynamic-ID type lookup should panic");
        let message = panic_message(panic);

        assert!(message.contains("has no static shard ID"));
        Ok(())
    }

    #[tokio::test]
    async fn get_shard_by_id_resolves_dynamic_id_shards() -> JaxResult<()> {
        let shard_id = test_id("00000000-0000-0000-0000-000000000402");
        let shard = Arc::new(DynamicIdShard(shard_id));
        let jax = Jax::default().register(shard.clone()).build()?;

        let resolved = jax.get_shard_by_id::<DynamicIdShard>(shard_id);

        assert!(Arc::ptr_eq(&resolved, &shard));
        Ok(())
    }

    #[tokio::test]
    async fn default_label_uses_rust_type_name() -> JaxResult<()> {
        let shard_id = test_id("00000000-0000-0000-0000-000000000405");
        let jax = Jax::default()
            .register(Arc::new(DefaultLabelShard))
            .build()?;

        let label = jax
            .list_shards()
            .into_iter()
            .find(|(id, _, _)| *id == shard_id)
            .map(|(_, label, _)| label)
            .expect("default label shard should be listed");

        assert_eq!(label, type_name::<DefaultLabelShard>());
        Ok(())
    }

    #[test]
    fn depends_macro_supports_mixed_types_and_ids() {
        let dependencies = MixedDependencyShard.dependencies();

        assert_eq!(
            dependencies,
            vec![
                test_id("00000000-0000-0000-0000-000000000301"),
                test_id("00000000-0000-0000-0000-000000000405"),
            ]
        );
    }

    fn panic_message(panic: Box<dyn core::any::Any + Send>) -> String {
        if let Some(message) = panic.downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else {
            "<non-string panic>".into()
        }
    }
}
