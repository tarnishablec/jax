#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use core::error::Error;

pub use jax_wasm_abi::{
    ABI_NAME, ABI_VERSION, FilesystemPermission, MANIFEST_CUSTOM_SECTION, PermissionManifest,
    ShardManifest, ShardManifestDependency,
};

pub type GuestError = Box<dyn Error + Send + Sync>;
pub type GuestResult<T> = Result<T, GuestError>;

pub struct Context;

impl Context {
    pub fn log(&self, _level: &str, _message: &str) {
        // Host logging import will be wired when the WIT bindings are implemented.
    }
}

pub trait Shard {
    fn setup(_context: Context) -> GuestResult<()> {
        Ok(())
    }

    fn teardown(_context: Context) -> GuestResult<()> {
        Ok(())
    }
}
