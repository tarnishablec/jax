#![no_std]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod timing;

pub use timing::TimingProbe;
