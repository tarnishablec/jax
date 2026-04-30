use crate::shard::{Shard, ShardId};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
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
        Self {
            graph,
            indices,
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
