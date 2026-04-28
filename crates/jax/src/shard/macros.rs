/// Implements the static identity metadata required by `TypedShard`.
///
/// # Example
/// ```
/// use jax::{shard_id, Shard, Descriptor, TypedShard};
///
/// struct MyShard;
///
/// impl TypedShard for MyShard {
///     shard_id!("67e55044-10b1-426f-9247-bb680e5fe0c8");
/// }
///
/// impl Shard for MyShard {
///     fn descriptor(&self) -> Descriptor {
///         Descriptor::typed::<Self>()
///     }
/// }
/// ```
#[macro_export]
macro_rules! shard_id {
    ($uuid:expr) => {
        fn static_id() -> $crate::ShardId
        where
            Self: Sized,
        {
            $crate::ShardId::from($crate::__private::uuid::uuid!($uuid))
        }
    };
}

#[macro_export]
macro_rules! depends {
    ($($shard:ty),* $(,)?) => {
        vec![
            $( $crate::Dependency::new(<$shard as $crate::TypedShard>::static_id()) ),*
        ]
    };
}
