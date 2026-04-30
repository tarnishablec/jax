use alloc::sync::Arc;
use jax::{Jax, JaxResult, Shard, shard_id};

extern crate alloc;

#[derive(Default)]
pub struct AddShard;

impl AddShard {
    pub fn add(&self, left: i32, right: i32) -> i32 {
        left + right
    }
}

#[async_trait::async_trait]
impl Shard for AddShard {
    shard_id!("2b30e84e-17eb-4ff9-ac96-40c9e40f8462");

    fn label(&self) -> &str {
        "example-add-shard"
    }

    async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        Ok(())
    }
}
