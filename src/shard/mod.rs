pub mod layer;
pub mod schedule;

use crate::Jax;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use async_trait::async_trait;
use core::any::type_name;
use core::error::Error;
use downcast_rs::{DowncastSync, impl_downcast};
use uuid::Uuid;

/// All feature shards must implement this trait.
///
/// Shards receive `Arc<Jax>` in `setup()` and can access other shards
/// via `jax.get_shard::<ConcreteType>()`.
#[async_trait]
pub trait Shard: DowncastSync + 'static {
    fn static_id() -> Uuid
    where
        Self: Sized;

    /// Stable UUID — must match the corresponding Web-side SHARD_IDS constant
    fn id(&self) -> Uuid;

    /// Human-readable display name (used for logging and get_shards response)
    fn label(&self) -> String {
        type_name::<Self>().into()
    }

    /// Called at app startup: subscribe to events, load config, etc.
    async fn setup(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Called at shard shutdown: unsubscribe from events, save config, etc.
    async fn teardown(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    fn dependencies(&self) -> Vec<Uuid> {
        vec![]
    }
}

impl_downcast!(sync Shard);

/// Implements the required identity metadata for a `Shard`.
///
/// This macro defines both the static and instance-level UUID methods required
/// by the `Jax` plugin system.
///
/// # Constraints
/// - The input must be a valid UUID string literal (e.g., `"550e8400-e29b-41d4-a716-446655440000"`).
/// - The implementation requires the `uuid` crate to be available in the scope.
///
/// # Example
/// ```
/// use jax::shard::Shard;
/// use jax::shard_id;
///
/// struct MyShard;
///
/// impl Shard for MyShard {
///     shard_id!("67e55044-10b1-426f-9247-bb680e5fe0c8");
///
///     // ... other trait methods
/// }
/// ```
#[macro_export]
macro_rules! shard_id {
    ($uuid:expr) => {
        fn static_id() -> uuid::Uuid
        where
            Self: Sized,
        {
            uuid::uuid!($uuid)
        }
        fn id(&self) -> uuid::Uuid {
            Self::static_id()
        }
    };
}

#[macro_export]
macro_rules! depends {
    ($($shard:ty),* $(,)?) => {
        vec![
            $( <$shard as $crate::Shard>::static_id() ),*
        ]
    };
}
