use std::cell::{BorrowError, BorrowMutError, Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Enables multiple owners (Rc) while allowing interior mutability (RefCell).
///
/// **Use-case**: Shared, mutable state in a single-threaded environment.
pub type RcRef<T> = Rc<RefCell<T>>;

/// Enables multiple owners (Arc) while ensuring safe, blocking interior mutability (Mutex).
///
/// Use-case: Shared, mutable state in a multi-threaded environment.
///           Every read *and* write must acquire the Mutex.
///           Prefer when you need mutable access often or the data is small.
pub type ArcRef<T> = Arc<Mutex<T>>;

/// Enables multiple owners (Arc) while allowing concurrent reads and exclusive writes (RwLock).
///
/// Use-case: Shared, mutable state in a multi-threaded environment where reads are frequent and writes are rare.
pub type ArcLock<T> = Arc<RwLock<T>>;

// pub type RcMut<T> = RcRef<T>;
// pub type ArcMut<T> = ArcRef<T>;
// pub type ARef<T> = ArcRef<T>;

pub trait Shared<T> {
    /// Shared (read-only) borrow.
    type Borrowed<'a>: Deref<Target = T>
    where
        Self: 'a;

    /// Exclusive (mutable) borrow.
    type BorrowedMut<'a>: DerefMut<Target = T>
    where
        Self: 'a;

    /// Error type for shared borrows.
    type BorrowError<'a>
    where
        Self: 'a;

    /// Error type for mutable borrows.
    type BorrowMutError<'a>
    where
        Self: 'a;
    fn new(t: T) -> Self;

    fn borrow(&self) -> Self::Borrowed<'_>;
    fn borrow_mut(&self) -> Self::BorrowedMut<'_>;

    fn with<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        f(&*self.borrow())
    }

    fn with_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.borrow_mut())
    }

    fn try_borrow(&self) -> Result<Self::Borrowed<'_>, Self::BorrowError<'_>>;
    fn try_borrow_mut(&self) -> Result<Self::BorrowedMut<'_>, Self::BorrowMutError<'_>>;

    fn try_with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, Self::BorrowError<'_>> {
        let guard = self.try_borrow()?;
        Ok(f(&*guard))
    }

    fn try_with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Result<R, Self::BorrowMutError<'_>> {
        let mut guard = self.try_borrow_mut()?;
        Ok(f(&mut *guard))
    }
}

/// Implementation for `RcRef<T>` = `Rc<RefCell<T>>` (single-threaded).
impl<T> Shared<T> for RcRef<T> {
    type Borrowed<'a>
        = Ref<'a, T>
    where
        Self: 'a;
    type BorrowedMut<'a>
        = RefMut<'a, T>
    where
        Self: 'a;

    // Std BorrowError / BorrowMutError do not depend on lifetimes,
    // but we can still use them as GATs by ignoring `'a`.
    type BorrowError<'a>
        = BorrowError
    where
        Self: 'a;
    type BorrowMutError<'a>
        = BorrowMutError
    where
        Self: 'a;

    fn new(t: T) -> Self {
        Rc::new(RefCell::new(t))
    }

    fn borrow(&self) -> Self::Borrowed<'_> {
        RefCell::borrow(self)
    }

    fn borrow_mut(&self) -> Self::BorrowedMut<'_> {
        RefCell::borrow_mut(self)
    }

    fn try_borrow(&self) -> Result<Self::Borrowed<'_>, Self::BorrowError<'_>> {
        self.as_ref().try_borrow()
    }

    fn try_borrow_mut(&self) -> Result<Self::BorrowedMut<'_>, Self::BorrowMutError<'_>> {
        self.as_ref().try_borrow_mut()
    }
}

/// Implementation for `ArcRef<T>` = `Arc<Mutex<T>>` (multi-threaded, all access behind Mutex).
impl<T> Shared<T> for ArcRef<T> {
    type Borrowed<'a>
        = MutexGuard<'a, T>
    where
        Self: 'a;
    type BorrowedMut<'a>
        = MutexGuard<'a, T>
    where
        Self: 'a;

    // Here the natural error is `PoisonError<MutexGuard<'a, T>>`,
    // exactly matching what `Mutex::lock` returns.
    type BorrowError<'a>
        = PoisonError<MutexGuard<'a, T>>
    where
        Self: 'a;
    type BorrowMutError<'a>
        = PoisonError<MutexGuard<'a, T>>
    where
        Self: 'a;

    fn new(t: T) -> Self {
        Arc::new(Mutex::new(t))
    }

    fn borrow(&self) -> Self::Borrowed<'_> {
        // `lock` can poison; unwrap to propagate panic if previous holder panicked.
        self.lock().expect("Mutex poisoned")
    }

    fn borrow_mut(&self) -> Self::BorrowedMut<'_> {
        self.lock().expect("Mutex poisoned")
    }

    fn try_borrow(&self) -> Result<Self::Borrowed<'_>, Self::BorrowError<'_>> {
        self.lock()
    }

    fn try_borrow_mut(&self) -> Result<Self::BorrowedMut<'_>, Self::BorrowMutError<'_>> {
        self.lock()
    }
}

/// Implementation for `ArcLock<T>` = `Arc<RwLock<T>>` (multi-threaded, read/write lock).
impl<T> Shared<T> for ArcLock<T> {
    type Borrowed<'a>
        = RwLockReadGuard<'a, T>
    where
        Self: 'a;
    type BorrowedMut<'a>
        = RwLockWriteGuard<'a, T>
    where
        Self: 'a;

    type BorrowError<'a>
        = PoisonError<RwLockReadGuard<'a, T>>
    where
        Self: 'a;
    type BorrowMutError<'a>
        = PoisonError<RwLockWriteGuard<'a, T>>
    where
        Self: 'a;

    fn new(t: T) -> Self {
        Arc::new(RwLock::new(t))
    }

    fn borrow(&self) -> Self::Borrowed<'_> {
        self.read().expect("RwLock poisoned")
    }

    fn borrow_mut(&self) -> Self::BorrowedMut<'_> {
        self.write().expect("RwLock poisoned")
    }

    fn try_borrow(&self) -> Result<Self::Borrowed<'_>, Self::BorrowError<'_>> {
        self.read()
    }

    fn try_borrow_mut(&self) -> Result<Self::BorrowedMut<'_>, Self::BorrowMutError<'_>> {
        self.write()
    }
}

pub fn shared<S, T>(t: T) -> S
where
    S: Shared<T>,
{
    S::new(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc_ref_basic() {
        let r: RcRef<i32> = shared(0);
        {
            let mut v = r.try_borrow_mut().unwrap();
            *v = 41;
        }
        assert_eq!(*r.try_borrow().unwrap(), 41);
    }

    #[test]
    fn arc_ref_basic() {
        let a: ArcRef<i32> = shared(10);
        {
            let mut v = a.try_borrow_mut().unwrap();
            *v += 1;
        }
        assert_eq!(*a.try_borrow().unwrap(), 11);
    }

    #[test]
    fn arc_lock_basic() {
        let a: ArcLock<i32> = shared(5);
        {
            let mut v = a.try_borrow_mut().unwrap();
            *v *= 2;
        }
        assert_eq!(*a.try_borrow().unwrap(), 10);
    }

    #[test]
    fn ext_helpers_scope_guards() {
        let v: ArcRef<i32> = shared(0);

        v.try_with_mut(|x| {
            *x = 123;
        })
        .unwrap();

        assert_eq!(*v.try_borrow().unwrap(), 123);
    }

    #[test]
    fn infallible_helpers() {
        let v: ArcRef<i32> = shared(1);

        v.with_mut(|x| {
            *x += 1;
        });

        v.with(|x| {
            assert_eq!(*x, 2);
        });
    }
}
