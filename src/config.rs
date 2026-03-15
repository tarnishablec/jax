/// Configuration for the Jax runtime.
#[derive(Debug, Clone)]
pub struct JaxConfig {
    /// Maximum number of concurrent shard setups within a single layer.
    /// Default is 50.
    pub max_concurrency: usize,
}

impl Default for JaxConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 50,
        }
    }
}
