use crate::shard::Shard;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use petgraph::prelude::*;
use uuid::Uuid;

pub(crate) type ShardGraph = StableDiGraph<Arc<dyn Shard>, ()>;

pub(crate) struct ShardRegistry {
    pub(crate) graph: ShardGraph,
    pub(crate) indices: BTreeMap<Uuid, NodeIndex>,
}
