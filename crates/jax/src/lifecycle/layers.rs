use crate::shard::Shard;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::error::Error;
use petgraph::prelude::*;

pub type ShardLayer = Vec<Arc<dyn Shard>>;

pub fn compute_shard_layers(
    graph: &StableDiGraph<Arc<dyn Shard>, ()>,
) -> Result<Vec<ShardLayer>, Box<dyn Error + Send + Sync>> {
    let mut layers = Vec::new();

    // 1. Compute in-degree for each node
    let mut in_degrees = BTreeMap::new();
    for node in graph.node_indices() {
        in_degrees.insert(node, graph.edges_directed(node, Incoming).count());
    }

    // 2. Collect all nodes with in-degree 0 (first layer)
    let mut current_layer_nodes: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|node| matches!(in_degrees.get(node), Some(0)))
        .collect();

    let mut processed_count = 0;

    // 3. Extract layers iteratively (Kahn's algorithm)
    while !current_layer_nodes.is_empty() {
        let mut next_layer_nodes = Vec::new();
        let mut current_layer_shards = Vec::with_capacity(current_layer_nodes.len());

        for &u in &current_layer_nodes {
            let shard = &graph[u];
            current_layer_shards.push(shard.clone());
            processed_count += 1;

            for edge in graph.edges_directed(u, Outgoing) {
                let v = edge.target();
                let Some(in_degree) = in_degrees.get_mut(&v) else {
                    return Err(alloc::format!(
                        "Jax: Missing in-degree state for shard node [{}]",
                        v.index()
                    )
                    .into());
                };

                *in_degree -= 1;

                if *in_degree == 0 {
                    next_layer_nodes.push(v);
                }
            }
        }

        layers.push(current_layer_shards);
        current_layer_nodes = next_layer_nodes;
    }

    // 4. Cycle detection: if processed count != total nodes, the graph contains a cycle
    if processed_count != graph.node_count() {
        return Err("Circular dependency detected! Some shards form a loop.".into());
    }

    Ok(layers)
}
