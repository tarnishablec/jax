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

pub use crate::error::{JaxError, JaxResult};
pub use crate::runtime::Jax;
pub use crate::shard::{Shard, ShardDependency, ShardDescriptor, TypedShard};
