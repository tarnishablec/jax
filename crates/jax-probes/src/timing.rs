use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use async_trait::async_trait;
use core::error::Error;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use jax::probe::Probe;
use jax::{Shard, ShardId};

/// A probe that records per-shard setup durations.
///
/// Requires a platform-specific clock function (`now_ns`) that returns
/// the current time in nanoseconds. This keeps the probe `no_std`-compatible.
///
/// # Example (with `std`)
/// ```ignore
/// use jax_probes::TimingProbe;
/// use std::time::Instant;
///
/// // Capture a reference point
/// let origin = Instant::now();
/// let timing = TimingProbe::new(move || origin.elapsed().as_nanos() as u64);
/// ```
pub struct TimingProbe {
    now_ns: Box<dyn Fn() -> u64 + Send + Sync>,
    /// Stores start timestamps keyed by shard index.
    /// Using a simple atomic slot per-shard would be ideal,
    /// but we use a fixed-size array approach via AtomicU64 map.
    starts: spin::RwLock<BTreeMap<ShardId, AtomicU64>>,
    durations: spin::RwLock<BTreeMap<ShardId, Duration>>,
}

impl TimingProbe {
    /// Create with a custom clock (for `no_std` or testing).
    pub fn with_clock(now_ns: impl Fn() -> u64 + Send + Sync + 'static) -> Self {
        Self {
            now_ns: Box::new(now_ns),
            starts: spin::RwLock::new(BTreeMap::new()),
            durations: spin::RwLock::new(BTreeMap::new()),
        }
    }

    /// Create with `std::time::Instant` as the clock source.
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        let origin = std::time::Instant::now();
        Self::with_clock(move || origin.elapsed().as_nanos() as u64)
    }

    /// Returns a snapshot of all recorded durations.
    pub fn durations(&self) -> BTreeMap<ShardId, Duration> {
        self.durations.read().clone()
    }
}

#[cfg(feature = "std")]
impl Default for TimingProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Probe for TimingProbe {
    async fn before_setup(&self, shard: &dyn Shard) -> Result<(), Box<dyn Error + Send + Sync>> {
        let now = (self.now_ns)();
        self.starts.write().insert(shard.id(), AtomicU64::new(now));
        Ok(())
    }

    async fn after_setup(
        &self,
        shard: &dyn Shard,
        _result: &Result<(), Box<dyn Error + Send + Sync>>,
    ) {
        let now = (self.now_ns)();
        let shard_id = shard.id();
        if let Some(start) = self.starts.read().get(&shard_id) {
            let start_ns = start.load(Ordering::Relaxed);
            let elapsed = Duration::from_nanos(now.saturating_sub(start_ns));
            self.durations.write().insert(shard_id, elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use core::sync::atomic::AtomicU64;
    use jax::{Jax, ShardId, depends, shard_id};
    use std::sync::Arc;

    struct FakeShard;

    #[async_trait]
    impl Shard for FakeShard {
        shard_id!("00000000-0000-0000-0000-000000000001");

        async fn setup(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
            // simulate ~50ms of work
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(())
        }
    }

    struct FailShard;

    #[async_trait]
    impl Shard for FailShard {
        shard_id!("00000000-0000-0000-0000-000000000002");

        async fn setup(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
            Err("boom".into())
        }
    }

    struct DepShard;

    #[async_trait]
    impl Shard for DepShard {
        shard_id!("00000000-0000-0000-0000-000000000003");

        depends![FakeShard];

        async fn setup(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(())
        }
    }

    struct AdvancingShard {
        counter: Arc<AtomicU64>,
    }

    #[async_trait]
    impl Shard for AdvancingShard {
        shard_id!("00000000-0000-0000-0000-000000000004");

        async fn setup(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn Error + Send + Sync>> {
            self.counter.fetch_add(50_000_000, Ordering::Relaxed);
            Ok(())
        }
    }

    fn test_id(input: &'static str) -> ShardId {
        input
            .parse()
            .unwrap_or_else(|error| panic!("invalid test shard id [{input}]: {error}"))
    }

    fn fake_clock() -> (
        impl Fn() -> u64 + Send + Sync + Clone + 'static,
        Arc<AtomicU64>,
    ) {
        let counter = Arc::new(AtomicU64::new(0));
        let c = counter.clone();
        let f = move || c.load(Ordering::Relaxed);
        (f, counter)
    }

    #[tokio::test]
    async fn records_setup_duration() {
        let (clock, counter) = fake_clock();
        let timing = Arc::new(TimingProbe::with_clock(clock));

        let jax = Jax::default()
            .probe(timing.clone())
            .register(Arc::new(AdvancingShard {
                counter: Arc::clone(&counter),
            }))
            .build()
            .expect("build failed");

        let jax = Arc::new(jax);
        counter.store(0, Ordering::Relaxed);
        let report = jax.start().await.expect("start failed");
        assert!(report.is_success());

        let durations = timing.durations();
        let elapsed = durations
            .get(&test_id("00000000-0000-0000-0000-000000000004"))
            .expect("duration not recorded");
        assert_eq!(*elapsed, Duration::from_millis(50));
    }

    #[tokio::test]
    async fn timing_probe_via_standalone_calls() {
        let (clock, counter) = fake_clock();
        let probe = TimingProbe::with_clock(clock);

        let shard = FakeShard;

        // Simulate before_setup at t=1000
        counter.store(1_000_000, Ordering::Relaxed);
        probe
            .before_setup(&shard as &dyn Shard)
            .await
            .expect("before_setup failed");

        // Simulate after_setup at t=2000 (1ms later)
        counter.store(2_000_000, Ordering::Relaxed);
        probe.after_setup(&shard as &dyn Shard, &Ok(())).await;

        let durations = probe.durations();
        let elapsed = durations
            .get(&test_id("00000000-0000-0000-0000-000000000001"))
            .expect("duration not recorded");
        assert_eq!(*elapsed, Duration::from_nanos(1_000_000));
    }

    #[tokio::test]
    async fn records_duration_on_failure() {
        let (clock, counter) = fake_clock();
        let probe = TimingProbe::with_clock(clock);

        let shard = FailShard;

        counter.store(0, Ordering::Relaxed);
        probe
            .before_setup(&shard as &dyn Shard)
            .await
            .expect("before_setup failed");

        counter.store(5_000_000, Ordering::Relaxed);
        let err: Result<(), Box<dyn Error + Send + Sync>> = Err("boom".into());
        probe.after_setup(&shard as &dyn Shard, &err).await;

        let durations = probe.durations();
        let elapsed = durations
            .get(&test_id("00000000-0000-0000-0000-000000000002"))
            .expect("duration not recorded for failed shard");
        assert_eq!(*elapsed, Duration::from_nanos(5_000_000));
    }

    #[tokio::test]
    async fn multiple_shards_tracked_independently() {
        let (clock, counter) = fake_clock();
        let probe = TimingProbe::with_clock(clock);

        let shard_a = FakeShard;
        let shard_b = FailShard;

        // shard_a: before at 0, after at 10ms
        counter.store(0, Ordering::Relaxed);
        probe
            .before_setup(&shard_a as &dyn Shard)
            .await
            .expect("ok");

        counter.store(10_000_000, Ordering::Relaxed);
        probe.after_setup(&shard_a as &dyn Shard, &Ok(())).await;

        // shard_b: before at 10ms, after at 13ms
        probe
            .before_setup(&shard_b as &dyn Shard)
            .await
            .expect("ok");

        counter.store(13_000_000, Ordering::Relaxed);
        probe.after_setup(&shard_b as &dyn Shard, &Ok(())).await;

        let durations = probe.durations();
        assert_eq!(durations.len(), 2);
        assert_eq!(
            *durations
                .get(&test_id("00000000-0000-0000-0000-000000000001"))
                .expect("missing"),
            Duration::from_nanos(10_000_000)
        );
        assert_eq!(
            *durations
                .get(&test_id("00000000-0000-0000-0000-000000000002"))
                .expect("missing"),
            Duration::from_nanos(3_000_000)
        );
    }

    #[tokio::test]
    async fn integration_with_dependency_chain() {
        let (clock, counter) = fake_clock();
        let timing = Arc::new(TimingProbe::with_clock(clock));

        let jax = Jax::default()
            .probe(timing.clone())
            .register(Arc::new(FakeShard))
            .register(Arc::new(DepShard))
            .build()
            .expect("build failed");

        let jax = Arc::new(jax);
        counter.store(0, Ordering::Relaxed);
        let report = jax.start().await.expect("start failed");
        assert!(report.is_success());

        let durations = timing.durations();
        assert_eq!(durations.len(), 2);
        assert!(durations.contains_key(&test_id("00000000-0000-0000-0000-000000000001")));
        assert!(durations.contains_key(&test_id("00000000-0000-0000-0000-000000000003")));
    }
}
