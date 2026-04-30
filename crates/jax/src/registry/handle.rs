use super::ShardRegistry;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use spin::Mutex;

/// Lock-protected handle for sharing immutable registry snapshots.
pub(crate) struct RegistryHandle {
    snapshot: Mutex<Option<Arc<ShardRegistry>>>,
    mutation: RegistryMutationLock,
}

impl Default for RegistryHandle {
    fn default() -> Self {
        Self {
            snapshot: Mutex::new(None),
            mutation: RegistryMutationLock::default(),
        }
    }
}

impl RegistryHandle {
    /// Snapshot of the current registry.
    pub(crate) fn snapshot(&self) -> Option<Arc<ShardRegistry>> {
        self.snapshot.lock().as_ref().map(Arc::clone)
    }

    /// Swap in a new immutable registry snapshot.
    pub(crate) fn swap(&self, new: Arc<ShardRegistry>) {
        *self.snapshot.lock() = Some(new);
    }

    /// Serialize async registry mutations while keeping read snapshots independent.
    pub(crate) fn lock_mutation(&self) -> RegistryMutationLockFuture<'_> {
        self.mutation.lock()
    }
}

#[derive(Default)]
pub(crate) struct RegistryMutationLock {
    state: Mutex<RegistryMutationState>,
}

#[derive(Default)]
struct RegistryMutationState {
    locked: bool,
    waiters: Vec<Waker>,
}

impl RegistryMutationLock {
    fn lock(&self) -> RegistryMutationLockFuture<'_> {
        RegistryMutationLockFuture { lock: self }
    }

    fn unlock(&self) {
        let waiters = {
            let mut state = self.state.lock();
            state.locked = false;
            mem::take(&mut state.waiters)
        };

        for waiter in waiters {
            waiter.wake();
        }
    }
}

pub(crate) struct RegistryMutationLockFuture<'a> {
    lock: &'a RegistryMutationLock,
}

impl<'a> Future for RegistryMutationLockFuture<'a> {
    type Output = RegistryMutationGuard<'a>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.lock.state.lock();

        if !state.locked {
            state.locked = true;
            return Poll::Ready(RegistryMutationGuard { lock: self.lock });
        }

        if !state
            .waiters
            .iter()
            .any(|waker| waker.will_wake(cx.waker()))
        {
            state.waiters.push(cx.waker().clone());
        }

        Poll::Pending
    }
}

pub(crate) struct RegistryMutationGuard<'a> {
    lock: &'a RegistryMutationLock,
}

impl Drop for RegistryMutationGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock();
    }
}
