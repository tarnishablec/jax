use crate::shard::{Shard, ShardId};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use petgraph::prelude::*;

pub(crate) type ShardGraph = StableDiGraph<Arc<dyn Shard>, ()>;

pub(crate) struct ShardRegistry {
    pub(crate) graph: ShardGraph,
    pub(crate) indices: BTreeMap<ShardId, NodeIndex>,
}
