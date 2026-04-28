//! Example guest shard.
//!
//! This file shows the target authoring model. The export macro and WIT binding
//! generation are not implemented yet.

use jax_wasm_abi::{PermissionManifest, ShardManifest};
use jax_wasm_guest::{Context, GuestResult, WasmShard};

pub struct ExampleGuestShard;

impl ExampleGuestShard {
    pub fn manifest() -> ShardManifest {
        ShardManifest::new(
            uuid::uuid!("550e8400-e29b-41d4-a716-446655440000"),
            "example-wasm-shard",
        )
        .with_permissions(PermissionManifest::new())
    }
}

impl WasmShard for ExampleGuestShard {
    fn setup(context: Context) -> GuestResult<()> {
        context.log("info", "example wasm shard setup");
        Ok(())
    }

    fn teardown(context: Context) -> GuestResult<()> {
        context.log("info", "example wasm shard teardown");
        Ok(())
    }
}
