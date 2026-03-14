use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use petgraph::prelude::*;

use crate::Shard;

/// A group of shards that have no dependencies on each other and can be initialized in parallel.
pub type ShardLayer = Vec<Arc<dyn Shard>>;

/// Computes a list of shard layers based on their dependency graph.
/// Uses Kahn's algorithm for topological sorting to group shards by execution depth.
pub fn compute_shard_layers(
    all_shards: &[Arc<dyn Shard>],
) -> Result<Vec<ShardLayer>, &'static str> {
    let n = all_shards.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // 1. Build the dependency graph
    let (graph, node_indices) = build_graph(all_shards)?;

    // 2. Compute the execution layers using topological sort
    let layer_indices = compute_layer_indices(&graph, &node_indices, n)?;

    // 3. Map the indices back to the actual Arc<dyn Shard> instances
    convert_indices_to_shards(all_shards, layer_indices)
}

/// Constructs a Directed Acyclic Graph (DAG) where nodes represent shards
/// and edges represent "Depends On" relationships.
fn build_graph(
    shards: &[Arc<dyn Shard>],
) -> Result<(DiGraph<(), ()>, Vec<NodeIndex>), &'static str> {
    let n = shards.len();
    let mut graph = DiGraph::with_capacity(n, n);

    let mut type_to_node = BTreeMap::new();
    let mut node_indices = Vec::with_capacity(n);

    // First pass: Add all shards as nodes and map their TypeId to their NodeIndex
    for shard in shards {
        let tid = shard.type_id();
        let idx = graph.add_node(());
        type_to_node.insert(tid, idx);
        node_indices.push(idx);
    }

    // Second pass: Add edges for each dependency
    for (i, shard) in shards.iter().enumerate() {
        for dep_tid in shard.dependencies() {
            if let Some(&dep_idx) = type_to_node.get(&dep_tid) {
                // Direction: Dependency -> Dependant (Source must be initialized before Target)
                graph.add_edge(dep_idx, node_indices[i], ());
            } else {
                return Err("Missing dependency: a shard depends on an unregistered type");
            }
        }
    }

    Ok((graph, node_indices))
}

/// Performs Kahn's Algorithm to partition the graph into sequential layers.
fn compute_layer_indices(
    graph: &DiGraph<(), ()>,
    node_indices: &[NodeIndex],
    total: usize,
) -> Result<Vec<Vec<NodeIndex>>, &'static str> {
    // Initialize in-degree counts (number of incoming edges for each node)
    let mut indegree = compute_indegrees(graph, node_indices, total);

    // Queue contains nodes with `0` in-degree (ready for initialization)
    let mut queue = init_zero_indegree_queue(node_indices, &indegree);

    let mut layers = Vec::new();
    let mut processed = 0usize;

    // Process the graph layer by layer
    while !queue.is_empty() {
        let current = process_one_layer(&mut queue, graph, &mut indegree, &mut processed);
        layers.push(current);
    }

    // If the number of processed nodes doesn't match the total, there is a cycle
    if processed != total {
        return Err("Circular dependency detected: shards cannot be ordered");
    }

    Ok(layers)
}

/// Calculates the in-degree for every node in the graph.
fn compute_indegrees(
    graph: &DiGraph<(), ()>,
    node_indices: &[NodeIndex],
    total: usize,
) -> Vec<usize> {
    let mut indegree = vec![0; total];
    for (i, &node) in node_indices.iter().enumerate() {
        indegree[i] = graph.neighbors_directed(node, Incoming).count();
    }
    indegree
}

/// Finds all nodes that have no prerequisites (in-degree of 0).
fn init_zero_indegree_queue(node_indices: &[NodeIndex], indegree: &[usize]) -> VecDeque<NodeIndex> {
    node_indices
        .iter()
        .copied()
        .enumerate()
        .filter(|&(idx, _)| indegree[idx] == 0)
        .map(|(_, node)| node)
        .collect()
}

/// Processes all currently ready nodes and returns them as a single layer.
/// Decrements the in-degree of their neighbors, potentially adding them to the queue.
fn process_one_layer(
    queue: &mut VecDeque<NodeIndex>,
    graph: &DiGraph<(), ()>,
    indegree: &mut [usize],
    processed: &mut usize,
) -> Vec<NodeIndex> {
    let size = queue.len();
    let mut layer = Vec::with_capacity(size);

    for _ in 0..size {
        let node = queue.pop_front().expect("Queue consistency error");

        layer.push(node);
        *processed += 1;

        // For each neighbor, reduce its in-degree as its dependency is now satisfied
        for neigh in graph.neighbors_directed(node, Outgoing) {
            let idx = neigh.index();
            indegree[idx] = indegree[idx].saturating_sub(1);
            if indegree[idx] == 0 {
                queue.push_back(neigh);
            }
        }
    }

    layer
}

/// Resolves the NodeIndex back to the original Arc<dyn Shard> pointer.
fn convert_indices_to_shards(
    all_shards: &[Arc<dyn Shard>],
    layers_indices: Vec<Vec<NodeIndex>>,
) -> Result<Vec<ShardLayer>, &'static str> {
    Ok(layers_indices
        .into_iter()
        .map(|layer_idx| {
            layer_idx
                .into_iter()
                .map(|node| all_shards[node.index()].clone())
                .collect()
        })
        .collect())
}
