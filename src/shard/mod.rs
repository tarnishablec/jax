pub mod dag;
pub mod info;

use crate::Jax;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use async_trait::async_trait;
use core::any::{TypeId, type_name};
use core::error::Error;
use heck::ToTitleCase;
use uuid::Uuid;

/// All feature shards must implement this trait.
///
/// Shards receive `Arc<Jax>` in `setup()` and can access other shards
/// via `jax.get_shard::<ConcreteType>()`.
#[async_trait]
pub trait Shard: Send + Sync {
    /// Stable UUID — must match the corresponding Web-side SHARD_IDS constant
    fn id(&self) -> Uuid;

    /// Human-readable display name (used for logging and get_shards response)
    fn label(&self) -> String {
        let full_name = type_name::<Self>();
        full_name
            .split("::")
            .last()
            .unwrap_or(full_name)
            .to_title_case()
    }

    /// Called at app startup: subscribe to events, load config, etc.
    async fn setup(&self, jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Called at shard shutdown: unsubscribe from events, save config, etc.
    async fn teardown(&self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn dependencies(&self) -> Vec<TypeId> {
        vec![]
    }
}

#[macro_export]
macro_rules! depends {
    ($($shard:ty),* $(,)?) => {
        vec![
            $( core::any::TypeId::of::<$shard>() ),*
        ]
    };
}
