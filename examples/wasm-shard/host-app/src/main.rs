//! Example host application loading a wasm component shard into Jax.

use example_wasm_typed_shard::LogShard;
use jax::Jax;
use jax_wasm::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let wasm_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_wasm_path);

    let jax = Jax::default()
        .with_wasm()
        .register(Arc::new(LogShard))
        .build()?;

    let report = jax.start().await?;
    assert!(report.is_success());

    let logger = jax.get_shard::<LogShard>();
    logger.log(Level::INFO, "host app started".into());

    let module = jax.load_wasm_from_file(&wasm_path).await?;
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

    let wasm_shard = jax.instantiate_wasm_with_config(&module, &config)?;
    let wasm_shard_id = wasm_shard.descriptor().id();
    jax.mount(wasm_shard).await?;
    logger.log(Level::INFO, "wasm shard mounted".into());
    jax.unmount(wasm_shard_id).await?;
    logger.log(Level::INFO, "wasm shard unmounted".into());

    jax.stop().await?;
    Ok(())
}

fn default_wasm_path() -> PathBuf {
    PathBuf::from("target/wasm32-wasip2/release/example_wasm_shard.wasm")
}
