use crate::shard::TypedShard;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use core::fmt;
use core::str::FromStr;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShardId(Uuid);

impl ShardId {
    pub fn parse_str(input: &str) -> Result<Self, uuid::Error> {
        input.parse()
    }
}

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Uuid> for ShardId {
    fn from(id: Uuid) -> Self {
        Self(id)
    }
}

impl FromStr for ShardId {
    type Err = uuid::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Ok(Self(input.parse()?))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Descriptor {
    /// Stable runtime identity used by the registry and dependency graph.
    id: ShardId,
    /// Human-readable runtime label for reports and diagnostics.
    label: String,
    /// Runtime dependencies that must be present before this shard can run.
    dependencies: Vec<Dependency>,
}

impl Descriptor {
    pub fn new(id: impl Into<ShardId>) -> Self {
        Self {
            id: id.into(),
            label: String::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn typed<T: TypedShard>() -> Self {
        Self::new(T::static_id()).with_label(type_name::<T>())
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_dependencies(mut self, dependencies: Vec<Dependency>) -> Self {
        self.dependencies = dependencies;
        self
    }

    pub fn id(&self) -> ShardId {
        self.id
    }

    pub fn label(&self) -> &str {
        if self.label.is_empty() {
            "<unnamed shard>"
        } else {
            &self.label
        }
    }

    pub fn dependencies(&self) -> &[Dependency] {
        &self.dependencies
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dependency {
    /// Stable runtime ID of the required provider shard.
    id: ShardId,
}

impl Dependency {
    pub fn new(id: impl Into<ShardId>) -> Self {
        Self { id: id.into() }
    }

    pub fn id(&self) -> ShardId {
        self.id
    }
}

impl From<Uuid> for Dependency {
    fn from(id: Uuid) -> Self {
        Self::new(id)
    }
}

impl From<ShardId> for Dependency {
    fn from(id: ShardId) -> Self {
        Self::new(id)
    }
}
