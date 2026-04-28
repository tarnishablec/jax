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

pub mod jax_wit {
    pub use crate::exports;

    pub mod runtime {
        pub use crate::jax::wit::runtime::*;
    }

    pub mod shard {
        pub use crate::exports::jax::wit::shard::*;
    }
}

pub mod wasm {
    pub use crate::Jax;
    pub use crate::WasmShard;
    pub use crate::jax_wit::shard::{Dependency, Descriptor};
}

pub type GuestError = Box<dyn Error + Send + Sync>;
pub type GuestResult<T> = Result<T, GuestError>;

pub struct Jax {
    _private: (),
}

static JAX: Jax = Jax { _private: () };

impl Jax {
    pub fn current() -> &'static Self {
        &JAX
    }

    pub fn call_raw(
        &self,
        shard_id: &str,
        method: &str,
        request_json: &str,
    ) -> Result<String, String> {
        jax_wit::runtime::call_shard(shard_id, method, request_json)
    }

    pub fn get_shard<'a, T>(&'a self) -> T
    where
        T: WasmShardClient<'a>,
    {
        T::from_jax(self)
    }
}

pub trait WasmShardClient<'a>: Sized {
    fn from_jax(jax: &'a Jax) -> Self;
}

pub trait WasmShard {
    type Config: DeserializeOwned + JsonSchema;

    fn describe() -> wasm::Descriptor;

    fn configure(_config: Self::Config) -> Result<(), String> {
        Ok(())
    }

    fn setup(_jax: &Jax) -> Result<(), String> {
        Ok(())
    }

    fn teardown(_jax: &Jax) -> Result<(), String> {
        Ok(())
    }
}

impl<T: WasmShard> crate::jax_wit::shard::Guest for T {
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
        T::setup(Jax::current())
    }

    fn teardown() -> Result<(), String> {
        T::teardown(Jax::current())
    }
}

#[macro_export]
macro_rules! export_shard {
    ($ty:ident) => {
        $crate::__export_shard!($ty);
    };
}
