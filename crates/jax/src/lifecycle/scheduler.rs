use crate::registry::ShardGraph;
use crate::shard::{Shard, ShardId};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::prelude::NodeIndex;

// Transient state bitmasks for one startup scheduling run.
pub const PENDING: u8 = 1 << 0;
pub const READY: u8 = 1 << 1;
pub const RUNNING: u8 = 1 << 2;
pub const SUCCESS: u8 = 1 << 3;
pub const FAILED: u8 = 1 << 4;
pub const SKIPPED: u8 = 1 << 5;
pub const CANCELLED: u8 = 1 << 7;

pub struct ShardScheduler<'a> {
    /// Read-only reference to the global plugin graph
    graph: &'a ShardGraph,
    /// Node-indexed atomic in-degree counters.
    in_degrees: BTreeMap<NodeIndex, AtomicUsize>,
    /// Node-indexed atomic bitmask states.
    states: BTreeMap<NodeIndex, AtomicU8>,
}

impl<'a> ShardScheduler<'a> {
    pub fn new(graph: &'a ShardGraph) -> Self {
        let mut in_degrees = BTreeMap::new();
        let mut states = BTreeMap::new();

        for idx in graph.node_indices() {
            let degree = graph.neighbors_directed(idx, Incoming).count();
            in_degrees.insert(idx, AtomicUsize::new(degree));
            states.insert(idx, AtomicU8::new(PENDING));
        }

        Self {
            graph,
            in_degrees,
            states,
        }
    }

    /// Extracts all seed nodes with in-degree 0
    pub fn collect_seeds(&self) -> Vec<(Arc<dyn Shard>, NodeIndex)> {
        self.graph
            .node_indices()
            .filter(|&idx| self.in_degree(idx).load(Ordering::Relaxed) == 0)
            .filter_map(|idx| {
                if self.try_activate(idx) {
                    Some((Arc::clone(&self.graph[idx]), idx))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Core logic: Upstream success
    pub fn notify_success(&self, idx: NodeIndex) -> Vec<(Arc<dyn Shard>, NodeIndex)> {
        // Mark as success and clear RUNNING/READY bits (optional)
        self.state(idx).fetch_or(SUCCESS, Ordering::Release);

        let mut newly_ready = Vec::new();

        for child_idx in self.graph.neighbors_directed(idx, Outgoing) {
            // Atomic subtraction: the last thread to complete dependencies is responsible for activation
            if self.in_degree(child_idx).fetch_sub(1, Ordering::AcqRel) == 1
                && self.try_activate(child_idx)
            {
                newly_ready.push((Arc::clone(&self.graph[child_idx]), child_idx));
            }
        }
        newly_ready
    }

    /// Core logic: Upstream failure
    pub fn notify_failure(&self, idx: NodeIndex) -> Vec<ShardId> {
        self.state(idx).fetch_or(FAILED, Ordering::Release);
        let mut skipped_ids = Vec::new();
        self.propagate_skip(idx, &mut skipped_ids);
        skipped_ids
    }

    /// Recursive pruning: mark all downstream as SKIPPED
    fn propagate_skip(&self, node_idx: NodeIndex, skipped: &mut Vec<ShardId>) {
        for child_idx in self.graph.neighbors_directed(node_idx, Outgoing) {
            let old = self.state(child_idx).fetch_or(SKIPPED, Ordering::AcqRel);

            // If not previously marked as SKIPPED, continue recursion
            if old & SKIPPED == 0 {
                skipped.push(self.graph[child_idx].id());
                self.propagate_skip(child_idx, skipped);
            }
        }
    }

    /// Status check: update RUNNING bit
    pub fn mark_running(&self, idx: NodeIndex) {
        self.state(idx).fetch_or(RUNNING, Ordering::Relaxed);
    }

    /// Atomic attempt to switch to execution state
    fn try_activate(&self, idx: NodeIndex) -> bool {
        let old = self.state(idx).fetch_or(READY, Ordering::AcqRel);
        // Only those not yet marked READY, and not marked SKIPPED or CANCELLED, can run
        (old & READY == 0) && (old & (SKIPPED | CANCELLED) == 0)
    }

    /// Get the final snapshot (used for generating reports)
    #[allow(dead_code)]
    pub fn get_status(&self, idx: NodeIndex) -> u8 {
        self.state(idx).load(Ordering::Acquire)
    }

    fn in_degree(&self, idx: NodeIndex) -> &AtomicUsize {
        self.in_degrees
            .get(&idx)
            .expect("Jax: scheduler missing in-degree for graph node")
    }

    fn state(&self, idx: NodeIndex) -> &AtomicU8 {
        self.states
            .get(&idx)
            .expect("Jax: scheduler missing state for graph node")
    }
}
