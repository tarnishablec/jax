extern crate alloc;

use alloc::sync::Arc;
use jax::{JaxResult, Shard, ShardDependency, ShardDescriptor};
use jax_wasm_abi::ShardManifest;

pub struct WasmShardLoader;

impl WasmShardLoader {
    pub fn new() -> Self {
        Self
    }

    pub async fn load_from_bytes(&self, _bytes: &[u8]) -> JaxResult<Arc<dyn Shard>> {
        Err("Jax WASM host: loading WASM shards is not implemented yet".into())
    }
}

impl Default for WasmShardLoader {
    fn default() -> Self {
        Self::new()
    }
}

pub fn descriptor_from_manifest(manifest: &ShardManifest) -> ShardDescriptor {
    let dependencies = manifest
        .dependencies()
        .iter()
        .map(|dependency| ShardDependency::new(dependency.id()))
        .collect();

    ShardDescriptor::new(manifest.id())
        .with_label(manifest.label())
        .with_dependencies(dependencies)
}
