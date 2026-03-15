use crate::shard::Shard;
use alloc::string::String;
use uuid::Uuid;

pub struct ShardInfo {
    pub id: Uuid,
    pub label: String,
}

pub trait ToInfo {
    fn info(&self) -> ShardInfo;
}

impl<T: Shard> ToInfo for T {
    fn info(&self) -> ShardInfo {
        ShardInfo {
            id: self.id(),
            label: self.label(),
        }
    }
}
