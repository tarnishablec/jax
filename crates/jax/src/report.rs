use crate::shard::ShardId;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::error::Error;

pub struct ShardError {
    pub id: ShardId,
    pub error: Box<dyn Error + Send + Sync>,
}

#[derive(Default)]
pub struct StartupReport {
    /// Shards that failed due to their own setup logic.
    pub failed: Vec<ShardError>,
    /// Shards that were skipped because their dependencies failed.
    pub skipped: Vec<ShardId>,
}

impl StartupReport {
    pub fn is_success(&self) -> bool {
        self.failed.is_empty() && self.skipped.is_empty()
    }
}

/// A clone-friendly summary of the startup report, stored for later queries.
/// Unlike `StartupReport`, this does not carry `Box<dyn Error>`.
#[derive(Default)]
pub struct StoredStartupReport {
    pub failed_ids: Vec<ShardId>,
    pub skipped: Vec<ShardId>,
}
