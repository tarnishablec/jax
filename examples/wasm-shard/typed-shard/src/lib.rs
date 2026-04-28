//! Example typed shard registered by the host application.

use jax::{Descriptor, Jax, JaxResult, Shard, TypedShard, shard_id};
use std::sync::{Arc, Once};
use tracing::Level;
use tracing_subscriber::filter::LevelFilter;

pub struct LogShard;

static TRACING_INIT: Once = Once::new();

impl LogShard {
    pub fn log(&self, level: Level, message: impl AsRef<str>) {
        let message = message.as_ref();
        match level {
            Level::TRACE => tracing::trace!(target: "example_wasm_typed_shard", "{}", message),
            Level::DEBUG => tracing::debug!(target: "example_wasm_typed_shard", "{}", message),
            Level::INFO => tracing::info!(target: "example_wasm_typed_shard", "{}", message),
            Level::WARN => tracing::warn!(target: "example_wasm_typed_shard", "{}", message),
            Level::ERROR => tracing::error!(target: "example_wasm_typed_shard", "{}", message),
        }
    }
}

impl TypedShard for LogShard {
    shard_id!("2b30e84e-17eb-4ff9-ac96-40c9e40f8462");
}

#[async_trait::async_trait]
impl Shard for LogShard {
    fn descriptor(&self) -> Descriptor {
        Descriptor::typed::<Self>().with_label("example-log-typed-shard")
    }

    async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        TRACING_INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_max_level(LevelFilter::INFO)
                .with_target(false)
                .try_init();
        });
        self.log(Level::INFO, "example typed shard setup");
        Ok(())
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        self.log(Level::INFO, "example typed shard teardown");
        Ok(())
    }
}
