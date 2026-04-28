use super::ShardRegistry;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicPtr, Ordering};

/// Atomic pointer wrapper for sharing immutable registry snapshots.
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

    /// Atomically swap in a new immutable registry snapshot.
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
