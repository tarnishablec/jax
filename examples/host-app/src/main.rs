use example_shards::AddShard;
use jax::Jax;
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let jax = Jax::default().register(Arc::new(AddShard)).build()?;
    let jax = Arc::new(jax);

    let report = jax.start().await?;
    assert!(report.is_success());

    let add = jax.get_shard::<AddShard>();
    let result = add.add(20, 22);
    assert_eq!(result, 42);
    println!("20 + 22 = {result}");

    jax.stop().await?;
    Ok(())
}
