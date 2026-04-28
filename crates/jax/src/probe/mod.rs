use crate::shard::Shard;
use alloc::boxed::Box;
use async_trait::async_trait;
use core::error::Error;

/// Non-invasive lifecycle observer for shards.
///
/// Probes are called before/after `setup()` and `teardown()` for every shard,
/// in registration order. They cannot modify shard behavior — only observe.
#[async_trait]
#[allow(unused_variables)]
pub trait Probe: Send + Sync + 'static {
    /// Called before a shard's `setup()`. Return `Err` to skip this shard.
    async fn before_setup(&self, shard: &dyn Shard) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Called after a shard's `setup()` completes (success or failure).
    async fn after_setup(
        &self,
        shard: &dyn Shard,
        result: &Result<(), Box<dyn Error + Send + Sync>>,
    ) {
    }

    /// Called before a shard's `teardown()`.
    async fn before_teardown(&self, shard: &dyn Shard) {}

    /// Called after a shard's `teardown()` completes.
    async fn after_teardown(
        &self,
        shard: &dyn Shard,
        result: &Result<(), Box<dyn Error + Send + Sync>>,
    ) {
    }
}
