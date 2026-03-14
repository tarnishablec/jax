#![no_std]
extern crate alloc;

pub mod shard;

use crate::shard::Shard;
use crate::shard::dag::compute_shard_layers;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use core::error::Error;
use core::sync::atomic::{AtomicPtr, Ordering};
use petgraph::stable_graph::StableDiGraph;

struct ShardRegistry {
    graph: StableDiGraph<Arc<dyn Shard>, ()>,
    indices: BTreeMap<TypeId, petgraph::graph::NodeIndex>,
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
    pub fn register<T: Shard + 'static>(&mut self, shard: Arc<T>) {
        self.pending.push(shard);
    }

    pub fn build(&mut self) {
        if (self.pending.is_empty()) {
            return;
        }
        let mut graph = StableDiGraph::new();
        let mut indices = BTreeMap::new();

        // 1. 提取所有暂存插件
        let shards: Vec<Arc<dyn Shard>> = self.pending.drain(..).collect();

        // 2. 映射所有节点
        for shard in &shards {
            let type_id = shard.type_id();
            let idx = graph.add_node(shard.clone());
            indices.insert(type_id, idx);
        }

        // 3. 构建依赖边
        for (type_id, &consumer_idx) in &indices {
            let shard = &graph[consumer_idx];
            for dep_id in shard.dependencies() {
                if let Some(&provider_idx) = indices.get(&dep_id) {
                    graph.add_edge(provider_idx, consumer_idx, ());
                }
            }
        }

        // 4. 循环依赖检测
        if petgraph::algo::is_cyclic_directed(&graph) {
            panic!("Jax: Circular dependency detected!");
        }

        // 5. 封装成新快照
        let new_registry = Arc::new(ShardRegistry { graph, indices });
        let new_ptr = Arc::into_raw(new_registry) as *mut ShardRegistry;

        // 6. 原子交换
        let old_ptr = self.registry.swap(new_ptr, Ordering::AcqRel);

        // 7. 处理旧内存
        if !old_ptr.is_null() {
            unsafe {
                Arc::from_raw(old_ptr); // 计数减 1，可能触发销毁
            }
        }
    }

    pub async fn start(self: &Arc<Self>) -> Result<(), Box<dyn Error + Send + Sync>> {
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
        let ptr = self.registry.load(Ordering::Acquire);
        let snapshot = unsafe {
            Arc::increment_strong_count(ptr);
            Arc::from_raw(ptr)
        };

        let node_idx = snapshot
            .indices
            .get(&TypeId::of::<T>())
            .expect("Jax: Shard not found.");
        let shard_dyn = snapshot.graph[*node_idx].clone();

        unsafe {
            let raw_dyn = Arc::into_raw(shard_dyn);
            let raw_t = raw_dyn as *const T;
            Arc::from_raw(raw_t)
        }
    }
}
