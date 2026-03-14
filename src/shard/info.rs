use crate::shard::Shard;
use alloc::string::String;
use uuid::Uuid;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
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
