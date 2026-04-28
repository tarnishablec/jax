pub use jax_wasm_contract as contract;
pub use jax_wasm_contract::*;
#[cfg(feature = "guest")]
pub use jax_wasm_guest as guest;
#[cfg(feature = "host")]
pub use jax_wasm_host as host;
pub use jax_wasm_macros::{export, import};

pub mod prelude {
    pub use schemars::JsonSchema;
    pub use serde::Deserialize;
    pub use serde_json::json;
    pub use tracing::Level;

    #[cfg(feature = "guest")]
    pub use crate::guest::wasm_shard;
    #[cfg(feature = "guest")]
    pub use crate::guest::wasm::{Dependency, Descriptor, Jax, WasmShard};
    #[cfg(feature = "host")]
    pub use crate::host::{WasmHostExports, WasmShardLoader};
    pub use crate::{export, import};
}
