use core::fmt;
use core::str::FromStr;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShardId(Uuid);

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Uuid> for ShardId {
    fn from(id: Uuid) -> Self {
        Self(id)
    }
}

impl From<&'static str> for ShardId {
    fn from(value: &'static str) -> Self {
        value
            .parse()
            .unwrap_or_else(|error| panic!("Jax: invalid shard id [{value}]: {error}"))
    }
}

impl FromStr for ShardId {
    type Err = uuid::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Ok(Self(input.parse()?))
    }
}
