/// Implements a shard's stable runtime ID from a UUID literal.
#[macro_export]
macro_rules! shard_id {
    ($uuid:literal) => {
        fn static_id() -> ::core::option::Option<$crate::ShardId>
        where
            Self: Sized,
        {
            ::core::option::Option::Some($crate::ShardId::from($crate::__private::uuid::uuid!(
                $uuid
            )))
        }

        fn id(&self) -> $crate::ShardId {
            <Self as $crate::Shard>::static_id()
                .unwrap_or_else(|| unreachable!("shard_id! always provides static_id"))
        }
    };
}

/// Implements a shard's dependencies from shard types and `ShardId` expressions.
#[macro_export]
macro_rules! depends {
    ($($tokens:tt)*) => {
        fn dependencies(&self) -> $crate::__private::Vec<$crate::ShardId> {
            $crate::__depends_vec!($($tokens)*)
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __depends_vec {
    ($($tokens:tt)*) => {{
        let mut dependencies = $crate::__private::Vec::new();
        $crate::__depends_push!(dependencies; $($tokens)*);
        dependencies
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __depends_push {
    ($dependencies:ident;) => {};
    ($dependencies:ident; , $($rest:tt)*) => {
        $crate::__depends_push!($dependencies; $($rest)*);
    };
    ($dependencies:ident; $shard:ty $(, $($rest:tt)*)?) => {
        $dependencies.push(
            <$shard as $crate::Shard>::static_id()
                .unwrap_or_else(|| panic!(
                    "Jax: dependency shard type [{}] has no static shard id",
                    ::core::any::type_name::<$shard>(),
                ))
        );
        $(
            $crate::__depends_push!($dependencies; $($rest)*);
        )?
    };
    ($dependencies:ident; $dependency:expr $(, $($rest:tt)*)?) => {
        let dependency: $crate::ShardId = $dependency;
        $dependencies.push(dependency);
        $(
            $crate::__depends_push!($dependencies; $($rest)*);
        )?
    };
}
