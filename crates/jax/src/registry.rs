use crate::shard::Shard;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicPtr, Ordering};
use petgraph::prelude::*;
use uuid::Uuid;

pub(crate) type ShardGraph = StableDiGraph<Arc<dyn Shard>, ()>;

pub(crate) struct ShardRegistry {
    pub(crate) graph: ShardGraph,
    pub(crate) native_indices: BTreeMap<Uuid, NodeIndex>,
    #[allow(dead_code)]
    pub(crate) guest_indices: BTreeMap<Uuid, NodeIndex>,
}

/// Atomic pointer wrapper for safely sharing `ShardRegistry` across threads.
pub(crate) struct RegistryHandle(AtomicPtr<ShardRegistry>);

impl Default for RegistryHandle {
    fn default() -> Self {
        Self(AtomicPtr::new(core::ptr::null_mut()))
    }
}

impl RegistryHandle {
    /// Atomically snapshot the current registry (increments Arc strong count).
    pub(crate) fn snapshot(&self) -> Option<Arc<ShardRegistry>> {
        let ptr = self.0.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        unsafe {
            Arc::increment_strong_count(ptr);
            Some(Arc::from_raw(ptr))
        }
    }

    /// Atomically swap in a new registry, returning the old raw pointer.
    pub(crate) fn swap(&self, new: Arc<ShardRegistry>) {
        let new_ptr = Arc::into_raw(new).cast_mut();
        let old_ptr = self.0.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            unsafe {
                drop(Arc::from_raw(old_ptr));
            }
        }
    }
}

impl Drop for RegistryHandle {
    fn drop(&mut self) {
        let ptr = self.0.swap(core::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            unsafe {
                drop(Arc::from_raw(ptr));
            }
        }
    }
}
