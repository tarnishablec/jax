/// Implements the static identity metadata required by `TypedShard`.
///
/// # Example
/// ```
/// use jax::{shard_id, Shard, ShardDescriptor, TypedShard};
///
/// struct MyShard;
///
/// impl TypedShard for MyShard {
///     shard_id!("67e55044-10b1-426f-9247-bb680e5fe0c8");
/// }
///
/// impl Shard for MyShard {
///     fn descriptor(&self) -> ShardDescriptor {
///         ShardDescriptor::typed::<Self>()
///     }
/// }
/// ```
#[macro_export]
macro_rules! shard_id {
    ($uuid:expr) => {
        fn static_id() -> uuid::Uuid
        where
            Self: Sized,
        {
            uuid::uuid!($uuid)
        }
    };
}

#[macro_export]
macro_rules! depends {
    ($($shard:ty),* $(,)?) => {
        vec![
            $( $crate::ShardDependency::new(<$shard as $crate::TypedShard>::static_id()) ),*
        ]
    };
}
