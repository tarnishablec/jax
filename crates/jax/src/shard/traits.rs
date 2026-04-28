use crate::{Jax, JaxResult};
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::sync::Arc;
use async_trait::async_trait;
use downcast_rs::{DowncastSync, impl_downcast};
use uuid::Uuid;

use super::ShardDescriptor;

/// A runtime shard that Jax can schedule and manage.
///
/// This trait intentionally models only lifecycle behavior and runtime
/// identity. Rust typed access is provided separately by `TypedShard`.
#[async_trait]
pub trait Shard: DowncastSync + Send + Sync + 'static {
    /// Returns stable runtime metadata for this shard.
    ///
    /// The descriptor's ID and dependency set must not change after the shard
    /// has been registered.
    fn descriptor(&self) -> ShardDescriptor;

    /// Called at app startup: subscribe to events, load config, etc.
    async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }

    /// Called at shard shutdown: unsubscribe from events, save config, etc.
    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }
}

impl_downcast!(sync Shard);

/// A shard whose runtime identity can be derived from a Rust type.
///
/// This powers `get_shard<T>()` and `depends![T]` without requiring every
/// runtime-loaded shard to have a host-side Rust type.
pub trait TypedShard: Shard {
    fn static_id() -> Uuid
    where
        Self: Sized;
}
