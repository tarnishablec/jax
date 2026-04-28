#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use serde_json::Value;
use tracing::Level;
use uuid::Uuid;

pub const ABI_NAME: &str = "jax-shard";
pub const ABI_VERSION: &str = "0.1.0";
pub const MANIFEST_CUSTOM_SECTION: &str = "jax.shard.manifest";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardManifest {
    /// Stable runtime identity for the shard contained in this artifact.
    id: Uuid,
    /// Human-readable label aligned with the runtime descriptor label.
    label: String,
    /// Runtime dependency IDs aligned with the core descriptor dependency model.
    dependencies: Vec<ShardManifestDependency>,
    /// Load-time capabilities requested by this wasm artifact.
    permissions: PermissionManifest,
}

impl ShardManifest {
    pub fn new(id: Uuid, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            dependencies: Vec::new(),
            permissions: PermissionManifest::default(),
        }
    }

    pub fn with_dependencies(mut self, dependencies: Vec<ShardManifestDependency>) -> Self {
        self.dependencies = dependencies;
        self
    }

    pub fn with_permissions(mut self, permissions: PermissionManifest) -> Self {
        self.permissions = permissions;
        self
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn dependencies(&self) -> &[ShardManifestDependency] {
        &self.dependencies
    }

    pub fn permissions(&self) -> &PermissionManifest {
        &self.permissions
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardManifestDependency {
    /// Stable runtime ID of another shard required by this artifact.
    id: Uuid,
}

impl ShardManifestDependency {
    pub fn new(id: Uuid) -> Self {
        Self { id }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PermissionManifest {
    /// Whether this artifact requests WASI host imports.
    wasi: bool,
    /// Filesystem access requested by this artifact.
    filesystem: FilesystemPermission,
    /// Whether this artifact requests outbound network access.
    network: bool,
}

impl PermissionManifest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_wasi(mut self, wasi: bool) -> Self {
        self.wasi = wasi;
        self
    }

    pub fn with_filesystem(mut self, filesystem: FilesystemPermission) -> Self {
        self.filesystem = filesystem;
        self
    }

    pub fn with_network(mut self, network: bool) -> Self {
        self.network = network;
        self
    }

    pub fn wasi(&self) -> bool {
        self.wasi
    }

    pub fn filesystem(&self) -> FilesystemPermission {
        self.filesystem
    }

    pub fn network(&self) -> bool {
        self.network
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FilesystemPermission {
    #[default]
    None,
    ReadOnly,
    ReadWrite,
}

pub trait WasmExport: Send + Sync + 'static {
    fn shard_id(&self) -> String;

    fn dispatch(&self, method: &str, request_json: &str) -> Result<String, String>;
}

pub trait ToWireValue {
    fn to_wire_value(&self) -> Result<Value, String>;
}

pub trait FromWireValue: Sized {
    fn from_wire_value(value: Value) -> Result<Self, String>;
}

pub trait IntoWireResult {
    fn into_wire_result(self) -> Result<String, String>;
}

impl ToWireValue for String {
    fn to_wire_value(&self) -> Result<Value, String> {
        Ok(Value::String(self.clone()))
    }
}

impl ToWireValue for Level {
    fn to_wire_value(&self) -> Result<Value, String> {
        Ok(Value::String(self.to_string()))
    }
}

impl FromWireValue for String {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        value
            .as_str()
            .map(ToString::to_string)
            .ok_or_else(|| "Jax WASM: expected string wire value".to_string())
    }
}

impl FromWireValue for Level {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        let input = String::from_wire_value(value)?;
        input
            .parse()
            .map_err(|error| alloc::format!("Jax WASM: invalid tracing level [{input}]: {error}"))
    }
}

impl IntoWireResult for () {
    fn into_wire_result(self) -> Result<String, String> {
        Ok("null".into())
    }
}

impl<T, E> IntoWireResult for Result<T, E>
where
    T: IntoWireResult,
    E: core::fmt::Display,
{
    fn into_wire_result(self) -> Result<String, String> {
        self.map_err(|error| error.to_string())?.into_wire_result()
    }
}

pub fn encode_request(values: Vec<Value>) -> Result<String, String> {
    serde_json::to_string(&values).map_err(|error| error.to_string())
}

pub fn decode_request(request_json: &str) -> Result<Vec<Value>, String> {
    serde_json::from_str(request_json).map_err(|error| error.to_string())
}

pub fn decode_response<T: FromWireValue>(response_json: &str) -> Result<T, String> {
    let value = serde_json::from_str(response_json).map_err(|error| error.to_string())?;
    T::from_wire_value(value)
}

impl FromWireValue for () {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        if value.is_null() {
            Ok(())
        } else {
            Err("Jax WASM: expected null wire response".into())
        }
    }
}
