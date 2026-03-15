# Jax

[![no_std compatible](https://img.shields.io/badge/no__std-compatible-seagreen.svg?style=flat-square)](https://crates.io/crates/jax)

Jax is a lightweight, dependency-aware plugin system for Rust applications. It allows you to register modular components called "shards" that have explicit dependencies and handle their lifecycle (setup and teardown) in the correct topological order with support for bounded concurrency and failure propagation.

## Features

- Declare shards with stable UUID identifiers and explicit dependencies
- Automatic dependency graph construction using [petgraph](https://github.com/petgraph/petgraph)
- Incremental registration and graph building
- Concurrent startup with bounded parallelism (configurable max concurrency)
- Dynamic topological execution using atomic in-degree tracking
- Automatic skipping of downstream shards on startup failure
- Graceful shutdown in reverse topological order (best-effort policy)
- Type-safe retrieval of native shards via `get_shard<T>()`
- no_std compatible (with alloc)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
jax = { git = "https://github.com/yourusername/jax.git", branch = "main" }  # or use a published version when available
```

## Quick Start
```rust
use jax::{Jax, JaxConfig};
use core::sync::Arc;

// Define your shard
struct DatabaseShard {
    // ...
}

impl DatabaseShard {
    fn new() -> Self {
        Self
    }
}

impl Shard for DatabaseShard {
    shard_id!("550e8400-e29b-41d4-a716-446655440000");

    async fn setup(&self, jax: Arc<Jax>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Initialize database connection, etc.
        Ok(())
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Close connection
        Ok(())
    }
}

struct AuthShard;

impl AuthShard {
    fn new() -> Self {
        Self
    }
}

impl Shard for AuthShard {
    shard_id!("67e55044-10b1-426f-9247-bb680e5fe0c8");

    fn dependencies(&self) -> Vec<Uuid> {
        depends![DatabaseShard]
    }

    async fn setup(&self, jax: Arc<Jax>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = jax.get_shard::<DatabaseShard>();
        // Use db to load auth data
        Ok(())
    }

    // ...
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn core::error::Error + Send + Sync>> {
    let jax = Jax::default()
        .register(Arc::new(DatabaseShard::new()))
        .register(Arc::new(AuthShard::new()))
        .build()?;

    let jax = Arc::new(jax);

    // Start all shards
    let report = jax.start().await?;
    println!("Startup report: failed={}, skipped={}", report.failed.len(), report.skipped.len());

    // Use shards
    let db = jax.get_shard::<DatabaseShard>();
    // ... use db

    // On shutdown
    jax.stop().await?;

    Ok(())
}
```
## License
Mozilla Public License Version 2.0