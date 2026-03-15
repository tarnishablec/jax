use crate::shard::Shard;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::prelude::NodeIndex;
use petgraph::stable_graph::StableDiGraph;

// --- State Bitmasks ---
pub const PENDING: u8 = 1 << 0;
pub const READY: u8 = 1 << 1;
pub const RUNNING: u8 = 1 << 2;
pub const SUCCESS: u8 = 1 << 3;
pub const FAILED: u8 = 1 << 4;
pub const SKIPPED: u8 = 1 << 5;
pub const TIMEOUT: u8 = 1 << 6;
pub const CANCELLED: u8 = 1 << 7;

pub struct ShardScheduler<'a> {
    /// Read-only reference to the global plugin graph
    graph: &'a StableDiGraph<Arc<dyn Shard>, ()>,
    /// Index-aligned atomic in-degree counters
    in_degrees: Vec<AtomicUsize>,
    /// Index-aligned atomic bitmask states
    states: Vec<AtomicU8>,
}

impl<'a> ShardScheduler<'a> {
    pub fn new(graph: &'a StableDiGraph<Arc<dyn Shard>, ()>) -> Self {
        let node_count = graph.node_count();
        let mut in_degrees = Vec::with_capacity(node_count);
        let mut states = Vec::with_capacity(node_count);

        for i in 0..node_count {
            let idx = NodeIndex::new(i);
            let degree = graph.neighbors_directed(idx, Incoming).count();
            in_degrees.push(AtomicUsize::new(degree));
            states.push(AtomicU8::new(PENDING));
        }

        Self {
            graph,
            in_degrees,
            states,
        }
    }

    /// Extracts all seed nodes with in-degree 0
    pub fn collect_seeds(&self) -> Vec<(Arc<dyn Shard>, usize)> {
        (0..self.graph.node_count())
            .filter(|&i| self.in_degrees[i].load(Ordering::Relaxed) == 0)
            .filter_map(|i| {
                if self.try_activate(i) {
                    Some((Arc::clone(&self.graph[NodeIndex::new(i)]), i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Core logic: Upstream success
    pub fn notify_success(&self, idx: usize) -> Vec<(Arc<dyn Shard>, usize)> {
        // Mark as success and clear RUNNING/READY bits (optional)
        self.states[idx].fetch_or(SUCCESS, Ordering::Release);

        let mut newly_ready = Vec::new();
        let node_idx = NodeIndex::new(idx);

        for child_idx in self.graph.neighbors_directed(node_idx, Outgoing) {
            let c_i = child_idx.index();
            // Atomic subtraction: the last thread to complete dependencies is responsible for activation
            if self.in_degrees[c_i].fetch_sub(1, Ordering::AcqRel) == 1 && self.try_activate(c_i) {
                newly_ready.push((Arc::clone(&self.graph[child_idx]), c_i));
            }
        }
        newly_ready
    }

    /// Core logic: Upstream failure
    pub fn notify_failure(&self, idx: usize) -> Vec<uuid::Uuid> {
        self.states[idx].fetch_or(FAILED, Ordering::Release);
        let mut skipped_ids = Vec::new();
        self.propagate_skip(NodeIndex::new(idx), &mut skipped_ids);
        skipped_ids
    }

    /// Recursive pruning: mark all downstream as SKIPPED
    fn propagate_skip(&self, node_idx: NodeIndex, skipped: &mut Vec<uuid::Uuid>) {
        for child_idx in self.graph.neighbors_directed(node_idx, Outgoing) {
            let c_i = child_idx.index();
            let old = self.states[c_i].fetch_or(SKIPPED, Ordering::AcqRel);

            // If not previously marked as SKIPPED, continue recursion
            if old & SKIPPED == 0 {
                skipped.push(self.graph[child_idx].id());
                self.propagate_skip(child_idx, skipped);
            }
        }
    }

    /// Status check: update RUNNING bit
    pub fn mark_running(&self, idx: usize) {
        self.states[idx].fetch_or(RUNNING, Ordering::Relaxed);
    }

    /// Atomic attempt to switch to execution state
    fn try_activate(&self, idx: usize) -> bool {
        let old = self.states[idx].fetch_or(READY, Ordering::AcqRel);
        // Only those not yet marked READY, and not marked SKIPPED or CANCELLED, can run
        (old & READY == 0) && (old & (SKIPPED | CANCELLED) == 0)
    }

    /// Get the final snapshot (used for generating reports)
    pub fn get_status(&self, idx: usize) -> u8 {
        self.states[idx].load(Ordering::Acquire)
    }
}
