//! Example guest shard implemented as a WebAssembly component.

use jax_wasm_guest::export_shard;
use jax_wasm_guest::wasm::{Descriptor, Jax, WasmShard};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::Level;

jax_wasm::import!(example_wasm_typed_shard::LogShard);

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExampleConfig {
    greeting: String,
    #[serde(default)]
    verbose: bool,
}

struct ExampleGuestShard;

impl WasmShard for ExampleGuestShard {
    type Config = ExampleConfig;

    fn describe() -> Descriptor {
        Descriptor {
            id: "550e8400-e29b-41d4-a716-446655440000".into(),
            label: "example-wasm-shard".into(),
            dependencies: Vec::new(),
        }
    }

    fn configure(config: Self::Config) -> Result<(), String> {
        let _ = config.greeting;
        let _ = config.verbose;
        Ok(())
    }

    fn setup(jax: &Jax) -> Result<(), String> {
        let logger = jax.get_shard::<LogShard>();
        logger.log(Level::INFO, "example WASM shard setup".into())?;
        Ok(())
    }

    fn teardown(jax: &Jax) -> Result<(), String> {
        let logger = jax.get_shard::<LogShard>();
        logger.log(Level::INFO, "example WASM shard teardown".into())?;
        Ok(())
    }
}

export_shard!(ExampleGuestShard);
