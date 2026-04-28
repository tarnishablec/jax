//! Example host application loading a wasm component shard into Jax.

use example_wasm_typed_shard::LogShard;
use jax::Jax;
use jax_wasm_host::WasmShardLoader;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::Level;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let wasm_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_wasm_path);

    let jax = Jax::default().register(Arc::new(LogShard)).build()?;
    let jax = Arc::new(jax);

    let report = jax.start().await?;
    assert!(report.is_success());

    let logger = jax.get_shard::<LogShard>();
    logger.log(Level::INFO, "host app started");

    let loader = WasmShardLoader::new()?;
    let module = loader.load_from_file(&wasm_path).await?;
    let schema = module.config_schema()?;
    jsonschema::meta::validate(&schema).map_err(|error| error.to_string())?;

    let config = json!({
        "greeting": "hello from host config",
        "verbose": true,
    });
    let validator = jsonschema::validator_for(&schema)?;
    validator
        .validate(&config)
        .map_err(|error| error.to_string())?;

    let wasm_shard = module.instantiate_with_config(&config)?;
    let wasm_shard_id = wasm_shard.descriptor().id();
    jax.mount(wasm_shard).await?;
    logger.log(Level::INFO, "wasm shard mounted");
    jax.unmount(wasm_shard_id).await?;
    logger.log(Level::INFO, "wasm shard unmounted");

    jax.stop().await?;
    Ok(())
}

fn default_wasm_path() -> PathBuf {
    PathBuf::from("target/wasm32-wasip2/release/example_wasm_shard.wasm")
}
