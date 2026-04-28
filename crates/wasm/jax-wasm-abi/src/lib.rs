#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
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
