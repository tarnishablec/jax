use crate::shard::{Shard, ShardId};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::TypeId;
use petgraph::prelude::*;
use spin::Mutex;

pub(crate) type ShardGraph = StableDiGraph<Arc<dyn Shard>, ()>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShardLifecycleState {
    Registered,
    Starting,
    Started,
    Failed,
    Skipped,
    Stopping,
    Stopped,
}

pub(crate) struct ShardRegistry {
    pub(crate) graph: ShardGraph,
    pub(crate) indices: BTreeMap<ShardId, NodeIndex>,
    pub(crate) type_index: BTreeMap<TypeId, Vec<ShardId>>,
    states: Mutex<BTreeMap<ShardId, ShardLifecycleState>>,
}

impl ShardRegistry {
    pub(crate) fn empty() -> Self {
        Self::new(StableDiGraph::new(), BTreeMap::new(), BTreeMap::new())
    }

    pub(crate) fn new(
        graph: ShardGraph,
        indices: BTreeMap<ShardId, NodeIndex>,
        states: BTreeMap<ShardId, ShardLifecycleState>,
    ) -> Self {
        let type_index = build_type_index(&graph, &indices);
        Self {
            graph,
            indices,
            type_index,
            states: Mutex::new(states),
        }
    }

    pub(crate) fn state(&self, shard_id: ShardId) -> Option<ShardLifecycleState> {
        self.states.lock().get(&shard_id).copied()
    }

    pub(crate) fn set_state(&self, shard_id: ShardId, state: ShardLifecycleState) {
        self.states.lock().insert(shard_id, state);
    }

    pub(crate) fn states_snapshot(&self) -> BTreeMap<ShardId, ShardLifecycleState> {
        self.states.lock().clone()
    }
}

fn build_type_index(
    graph: &ShardGraph,
    indices: &BTreeMap<ShardId, NodeIndex>,
) -> BTreeMap<TypeId, Vec<ShardId>> {
    let mut type_index = BTreeMap::new();
    for (&shard_id, &node_idx) in indices {
        type_index
            .entry(graph[node_idx].as_ref().type_id())
            .or_insert_with(Vec::new)
            .push(shard_id);
    }
    type_index
}
