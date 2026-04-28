//! Example guest shard implemented as a WebAssembly component.

use jax_wasm_guest::export_shard;
use jax_wasm_guest::wasm::{self, Descriptor, WasmShard};
use schemars::JsonSchema;
use serde::Deserialize;

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
        if config.verbose {
            wasm::host::log(
                "info",
                &format!("example WASM shard configured: {}", config.greeting),
            );
        }
        Ok(())
    }

    fn setup() -> Result<(), String> {
        wasm::host::log("info", "example WASM shard setup");
        Ok(())
    }

    fn teardown() -> Result<(), String> {
        wasm::host::log("info", "example WASM shard teardown");
        Ok(())
    }
}

export_shard!(ExampleGuestShard);
