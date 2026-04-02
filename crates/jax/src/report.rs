use alloc::boxed::Box;
use alloc::vec::Vec;
use core::error::Error;
use uuid::Uuid;

pub struct ShardError {
    pub id: Uuid,
    pub error: Box<dyn Error + Send + Sync>,
}

#[derive(Default)]
pub struct StartupReport {
    /// Shards that failed due to their own setup logic.
    pub failed: Vec<ShardError>,
    /// Shards that were skipped because their dependencies failed.
    pub skipped: Vec<Uuid>,
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
    pub failed_ids: Vec<Uuid>,
    pub skipped: Vec<Uuid>,
}
