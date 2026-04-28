use crate::shard::TypedShard;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use uuid::Uuid;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardDescriptor {
    id: Uuid,
    label: String,
    dependencies: Vec<ShardDependency>,
}

impl ShardDescriptor {
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
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

    pub fn with_dependencies(mut self, dependencies: Vec<ShardDependency>) -> Self {
        self.dependencies = dependencies;
        self
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn label(&self) -> &str {
        if self.label.is_empty() {
            "<unnamed shard>"
        } else {
            &self.label
        }
    }

    pub fn dependencies(&self) -> &[ShardDependency] {
        &self.dependencies
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardDependency {
    id: Uuid,
}

impl ShardDependency {
    pub fn new(id: Uuid) -> Self {
        Self { id }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }
}

impl From<Uuid> for ShardDependency {
    fn from(id: Uuid) -> Self {
        Self::new(id)
    }
}
