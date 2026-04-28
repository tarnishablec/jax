#![no_std]
extern crate alloc;

pub mod config;
pub mod error;
mod lifecycle;
pub mod probe;
pub(crate) mod registry;
pub mod report;
mod runtime;
pub mod shard;

#[doc(hidden)]
pub mod __private {
    pub use uuid;
}

pub use crate::error::{JaxError, JaxResult};
pub use crate::runtime::Jax;
pub use crate::shard::{Dependency, Descriptor, Shard, ShardId, TypedShard};
