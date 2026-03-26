//! Type-map metadata container.
//!
//! [`TypedMetadata`] stores arbitrary `Send + Sync + 'static` values keyed by
//! their concrete [`TypeId`]. At most one value per concrete type. This is the
//! standard "type-map" or "AnyMap" pattern.
//!
//! Used on `FrameEnvelope`, `Detection`, `Track`, `OutputEnvelope`,
//! and `PerceptionArtifacts` to carry extensible domain-specific data
//! without modifying core types.
//!
//! # Cloning cost
//!
//! `TypedMetadata::clone()` deep-clones every stored value. Metadata bags are
//! typically small (2–5 entries) with lightweight types. If a stage stores large
//! data (e.g., a full feature map), wrap it in `Arc<T>` so that cloning the bag
//! clones only the `Arc`, not the data.

use std::any::{Any, TypeId};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal cloneable entry — stores a type-erased value together with a
// function pointer that knows how to deep-clone it.  This avoids a custom
// `CloneableAny` trait whose `as_any` return trips borrow-checker lifetime
// issues.
// ---------------------------------------------------------------------------

struct CloneableEntry {
    value: Box<dyn Any + Send + Sync>,
    clone_fn: fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>,
}

fn make_clone_fn<T: Clone + Send + Sync + 'static>()
-> fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync> {
    |any| {
        // Safety-net: downcast must succeed because the clone_fn is monomorphised
        // for the same T that was inserted. A failure here indicates memory
        // corruption or a logic bug — log and abort rather than panicking with
        // an opaque message in production.
        let val = match any.downcast_ref::<T>() {
            Some(v) => v,
            None => {
                // This branch is structurally unreachable: the clone_fn is
                // always paired with the correct type at insertion time.
                // If it ever fires, something is deeply wrong.
                unreachable!(
                    "TypedMetadata clone: type mismatch for TypeId {:?}",
                    std::any::TypeId::of::<T>()
                );
            }
        };
        Box::new(val.clone())
    }
}

impl Clone for CloneableEntry {
    fn clone(&self) -> Self {
        Self {
            value: (self.clone_fn)(&*self.value),
            clone_fn: self.clone_fn,
        }
    }
}

/// Typed metadata bag — stores arbitrary `Send + Sync + 'static` values
/// keyed by their concrete type.
///
/// At most one value per concrete type. To store multiple values of the same
/// underlying type, use distinct newtype wrappers.
///
/// # Example
///
/// ```
/// use nv_core::TypedMetadata;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct DetectorScore(f32);
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct TrackerScore(f32);
///
/// let mut meta = TypedMetadata::new();
/// meta.insert(DetectorScore(0.95));
/// meta.insert(TrackerScore(0.8));
///
/// assert_eq!(meta.get::<DetectorScore>(), Some(&DetectorScore(0.95)));
/// assert_eq!(meta.get::<TrackerScore>(), Some(&TrackerScore(0.8)));
/// assert_eq!(meta.len(), 2);
/// ```
pub struct TypedMetadata {
    map: HashMap<TypeId, CloneableEntry>,
}

impl TypedMetadata {
    /// Create an empty metadata bag.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a value. If a value of this type already exists, it is replaced
    /// and the old value is returned.
    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, val: T) -> Option<T> {
        let entry = CloneableEntry {
            value: Box::new(val),
            clone_fn: make_clone_fn::<T>(),
        };
        self.map
            .insert(TypeId::of::<T>(), entry)
            .and_then(|old| old.value.downcast::<T>().ok())
            .map(|b| *b)
    }

    /// Get a reference to the stored value of type `T`, if present.
    #[must_use]
    pub fn get<T: 'static>(&self) -> Option<&T> {
        let entry = self.map.get(&TypeId::of::<T>())?;
        entry.value.downcast_ref()
    }

    /// Get a mutable reference to the stored value of type `T`, if present.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let entry = self.map.get_mut(&TypeId::of::<T>())?;
        entry.value.downcast_mut()
    }

    /// Remove and return the stored value of type `T`, if present.
    pub fn remove<T: 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|old| old.value.downcast::<T>().ok())
            .map(|b| *b)
    }

    /// Returns `true` if a value of type `T` is stored.
    #[must_use]
    pub fn contains<T: 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the metadata bag is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Merge another `TypedMetadata` into this one.
    ///
    /// Keys present in `other` overwrite keys in `self` (last-writer-wins).
    pub fn merge(&mut self, other: TypedMetadata) {
        self.map.extend(other.map);
    }
}

impl Clone for TypedMetadata {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl Default for TypedMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TypedMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypedMetadata")
            .field("len", &self.map.len())
            .finish()
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Foo(u32);

    #[derive(Clone, Debug, PartialEq)]
    struct Bar(String);

    #[test]
    fn insert_and_get() {
        let mut m = TypedMetadata::new();
        m.insert(Foo(42));
        assert_eq!(m.get::<Foo>(), Some(&Foo(42)));
        assert_eq!(m.get::<Bar>(), None);
    }

    #[test]
    fn replace_returns_old() {
        let mut m = TypedMetadata::new();
        assert!(m.insert(Foo(1)).is_none());
        let old = m.insert(Foo(2));
        assert_eq!(old, Some(Foo(1)));
        assert_eq!(m.get::<Foo>(), Some(&Foo(2)));
    }

    #[test]
    fn remove() {
        let mut m = TypedMetadata::new();
        m.insert(Foo(10));
        let removed = m.remove::<Foo>();
        assert_eq!(removed, Some(Foo(10)));
        assert!(!m.contains::<Foo>());
    }

    #[test]
    fn clone_is_deep() {
        let mut m = TypedMetadata::new();
        m.insert(Bar("hello".into()));
        let m2 = m.clone();
        m.insert(Bar("changed".into()));
        assert_eq!(m2.get::<Bar>(), Some(&Bar("hello".into())));
    }

    #[test]
    fn merge_overwrites() {
        let mut a = TypedMetadata::new();
        a.insert(Foo(1));
        let mut b = TypedMetadata::new();
        b.insert(Foo(2));
        b.insert(Bar("from_b".into()));
        a.merge(b);
        assert_eq!(a.get::<Foo>(), Some(&Foo(2)));
        assert_eq!(a.get::<Bar>(), Some(&Bar("from_b".into())));
    }
}
