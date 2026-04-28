#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use core::error::Error;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;

pub use jax_wasm_abi::{
    ABI_NAME, ABI_VERSION, FilesystemPermission, MANIFEST_CUSTOM_SECTION, PermissionManifest,
    ShardManifest, ShardManifestDependency,
};

wit_bindgen::generate!({
    world: "jax-shard",
    path: "../jax-wasm-abi/wit",
    export_macro_name: "__export_shard",
    pub_export_macro: true,
    default_bindings_module: "jax_wasm_guest",
});

pub mod wasm {
    pub mod host {
        pub use crate::jax::wasm::host::log;
    }

    pub use crate::WasmShard;
    pub use crate::exports::jax::wasm::shard::{Dependency, Descriptor};
}

pub type GuestError = Box<dyn Error + Send + Sync>;
pub type GuestResult<T> = Result<T, GuestError>;

pub trait WasmShard {
    type Config: DeserializeOwned + JsonSchema;

    fn describe() -> wasm::Descriptor;

    fn configure(_config: Self::Config) -> Result<(), String> {
        Ok(())
    }

    fn setup() -> Result<(), String> {
        Ok(())
    }

    fn teardown() -> Result<(), String> {
        Ok(())
    }
}

impl<T: WasmShard> exports::jax::wasm::shard::Guest for T {
    fn describe() -> wasm::Descriptor {
        T::describe()
    }

    fn config_schema() -> String {
        serde_json::to_string(&schema_for!(T::Config))
            .unwrap_or_else(|error| alloc::format!(r#"{{"error":"{error}"}}"#))
    }

    fn configure(config: String) -> Result<(), String> {
        let config = serde_json::from_str(&config).map_err(|error| error.to_string())?;
        T::configure(config)
    }

    fn setup() -> Result<(), String> {
        T::setup()
    }

    fn teardown() -> Result<(), String> {
        T::teardown()
    }
}

#[macro_export]
macro_rules! export_shard {
    ($ty:ident) => {
        $crate::__export_shard!($ty);
    };
}
