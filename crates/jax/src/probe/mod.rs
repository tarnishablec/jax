use crate::{JaxResult, shard::Shard};
#[allow(unused_imports)]
use alloc::boxed::Box;
use async_trait::async_trait;

/// Non-invasive lifecycle observer for shards.
///
/// Probes are called before/after `setup()` and `teardown()` for every shard,
/// in registration order. They cannot modify shard behavior, only observe.
#[async_trait]
#[allow(unused_variables)]
pub trait Probe: Send + Sync + 'static {
    /// Called before a shard's `setup()`.
    async fn before_setup(&self, shard: &dyn Shard) {}

    /// Called after a shard's `setup()` completes (success or failure).
    async fn after_setup(&self, shard: &dyn Shard, result: &JaxResult<()>) {}

    /// Called before a shard's `teardown()`.
    async fn before_teardown(&self, shard: &dyn Shard) {}

    /// Called after a shard's `teardown()` completes.
    async fn after_teardown(&self, shard: &dyn Shard, result: &JaxResult<()>) {}
}
