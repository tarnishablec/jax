use crate::{Jax, JaxResult};
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use async_trait::async_trait;
use core::any::type_name;
use downcast_rs::{DowncastSync, impl_downcast};

use super::ShardId;

/// A runtime shard that Jax can schedule and manage.
///
/// This trait intentionally models only lifecycle behavior and the minimal
/// metadata needed to schedule that lifecycle.
#[async_trait]
pub trait Shard: DowncastSync + Send + Sync + 'static {
    /// Optional Rust type-level ID for ergonomic static references.
    ///
    /// Runtime identity still comes from `id(&self)`. Shards with dynamic or
    /// instance-specific IDs can leave this as `None` and implement `id`
    /// directly.
    fn static_id() -> Option<ShardId>
    where
        Self: Sized,
    {
        None
    }

    /// Stable runtime identity used by the registry and dependency graph.
    ///
    /// The ID must not change after the shard has been registered.
    fn id(&self) -> ShardId;

    /// Human-readable runtime label for reports and diagnostics.
    fn label(&self) -> &str {
        type_name::<Self>()
    }

    /// Runtime dependencies that must be present before this shard can run.
    ///
    /// The dependency set must not change after the shard has been registered.
    fn dependencies(&self) -> Vec<ShardId> {
        vec![]
    }

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
