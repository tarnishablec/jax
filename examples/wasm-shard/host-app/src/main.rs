//! Example host application.
//!
//! This file shows the intended host-side shape. `load_from_bytes` currently
//! returns an unimplemented error until the wasm component loader is wired.

use jax::Jax;
use jax_wasm_host::WasmShardLoader;
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let loader = WasmShardLoader::new();
    let bytes = std::fs::read(
        "../guest-shard/target/wasm32-unknown-unknown/release/example_wasm_shard.wasm",
    )?;
    let shard = loader.load_from_bytes(&bytes).await?;

    let jax = Jax::default().build()?;
    let jax = Arc::new(jax);

    // Future core API:
    // jax.load_shard(shard).await?;
    drop(shard);

    let report = jax.start().await?;
    assert!(report.is_success());

    jax.stop().await?;
    Ok(())
}
