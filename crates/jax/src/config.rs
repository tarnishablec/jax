use crate::probe::Probe;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Configuration for the Jax runtime.
pub struct JaxConfig {
    /// Maximum number of concurrent shard setups within a single layer.
    /// Default is 50.
    pub max_concurrency: usize,
    /// Registered lifecycle probes, called in order.
    pub(crate) probes: Vec<Arc<dyn Probe>>,
}

impl Default for JaxConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 50,
            probes: Vec::new(),
        }
    }
}
