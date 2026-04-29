# Jax

[![no_std compatible](https://img.shields.io/badge/no__std-compatible-seagreen.svg?style=flat-square)](https://crates.io/crates/jax)

Jax is a lightweight, dependency-aware lifecycle runtime for Rust applications.
It manages opaque shards with stable IDs, explicit dependencies, and ordered
setup/teardown. Cross-runtime guest protocols, exports, and invocation routing
are intentionally outside `jax` core.

## Features

- Declare shards with stable UUID identifiers and explicit dependencies
- Automatic dependency graph construction using [petgraph](https://github.com/petgraph/petgraph)
- Incremental registration and graph building
- Concurrent startup with bounded parallelism
- Dynamic topological execution using atomic in-degree tracking
- Automatic skipping of downstream shards on startup failure
- Graceful shutdown in reverse topological order
- Runtime mount/unmount of opaque `Shard` instances
- Concrete Rust type lookup via `get_shard<T>()` when exactly one instance of `T` is registered
- no_std compatible (with alloc)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
jax = { git = "https://github.com/tarnishablec/jax.git", branch = "main" }  # or use a published version when available
async-trait = "0.1"
uuid = "1"
```

## Quick Start

```rust
use std::sync::Arc;
use jax::{depends, shard_id, Jax, JaxResult, Shard};

struct DatabaseShard;

impl DatabaseShard {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Shard for DatabaseShard {
    shard_id!("550e8400-e29b-41d4-a716-446655440000");

    fn label(&self) -> &str {
        "database"
    }

    async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }
}

struct AuthShard;

impl AuthShard {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Shard for AuthShard {
    shard_id!("67e55044-10b1-426f-9247-bb680e5fe0c8");

    fn label(&self) -> &str {
        "auth"
    }

    depends![DatabaseShard];

    async fn setup(&self, jax: Arc<Jax>) -> JaxResult<()> {
        let _db = jax.get_shard::<DatabaseShard>();
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn core::error::Error + Send + Sync>> {
    let jax = Jax::default()
        .register(Arc::new(DatabaseShard::new()))
        .register(Arc::new(AuthShard::new()))
        .build()?;

    let jax = Arc::new(jax);

    let report = jax.start().await?;
    println!(
        "Startup report: failed={}, skipped={}",
        report.failed.len(),
        report.skipped.len()
    );

    let _db = jax.get_shard::<DatabaseShard>();

    jax.stop().await?;

    Ok(())
}
```

## License

Mozilla Public License Version 2.0
