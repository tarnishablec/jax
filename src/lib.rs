#![no_std]
extern crate alloc;

pub mod shard;

use crate::shard::Shard;
use crate::shard::dag::compute_shard_layers;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use core::error::Error;

/// Core application context.
///
/// Lifecycle: `new()` → `register()` → `Arc::new()` → `start()`.
/// Tauri drives it: `Builder::setup()` builds and starts Jax.
#[derive(Default)]
pub struct Jax {
    shards: Vec<Arc<dyn Shard>>,
    typed: BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Jax {
    /// Register a shard. Must be called before `start()`.
    pub fn register<T: Shard + 'static>(&mut self, shard: Arc<T>) {
        let tid = TypeId::of::<T>();

        self.shards.push(shard.clone() as Arc<dyn Shard>);
        self.typed.insert(tid, shard);

        tracing::debug!("Registered shard: {}", core::any::type_name::<T>());
    }

    pub fn start(self: &Arc<Self>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let layers = compute_shard_layers(&self.shards)?;

        for layer in layers {
            let mut handles = Vec::new();

            for shard in layer {
                let jax = self.clone();
                let shard_cloned = shard.clone();

                let handle =
                    tauri::async_runtime::spawn(async move { shard_cloned.setup(jax).await });
                handles.push(handle);
            }

            tauri::async_runtime::block_on(async {
                for handle in handles {
                    handle.await.map_err(|_| "Task join failed")??;
                }
                Ok::<(), Box<dyn Error + Send + Sync>>(())
            })?;

            tracing::debug!("Shard layer completed, moving to next layer...");
        }

        tracing::info!("Jax: All shards started successfully.");
        Ok(())
    }

    /// Retrieve a shard by its concrete type.
    ///
    /// ```ignore
    /// let lcu = jax.get_shard::<LcuShard>();
    /// lcu.subscribe();
    /// ```
    ///
    /// Panics if `T` was never registered.
    pub fn get_shard<T: Shard + Send + Sync + 'static>(&self) -> Arc<T> {
        self.typed
            .get(&TypeId::of::<T>())
            .and_then(|any| any.clone().downcast::<T>().ok())
            .expect("shard not registered")
    }
}
