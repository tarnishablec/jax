use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::error::Error;
use core::time::Duration;
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
    /// Per-shard setup durations (populated when `setup-timing` feature is enabled).
    pub durations: BTreeMap<Uuid, Duration>,
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
    pub durations: BTreeMap<Uuid, Duration>,
}
