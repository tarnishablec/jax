# Wasm Shard Example

This example shows the intended end-to-end shape for a Rust-authored wasm shard.

The wasm crates are still scaffolding, so this example is documentation-first:
it describes the target authoring flow without being part of the workspace build.

## Layout

```text
guest-shard/
  A Rust crate compiled to a wasm component.

host-app/
  A Rust application that loads the wasm shard and registers it into Jax.
```

## Intended Flow

1. The guest shard declares stable metadata through a manifest.
2. The guest shard exports lifecycle functions described by `jax-shard.wit`.
3. The host loader reads the wasm file, validates the manifest, and creates an `Arc<dyn jax::Shard>`.
4. The application gives that shard to Jax.
5. Jax schedules it like any other shard by UUID dependencies.

## Boundary

`jax` core does not know about wasm. The host adapter is responsible for turning
a wasm component into `Arc<dyn jax::Shard>`.

For the current builder-only core, a wasm shard would need to be loaded before
`Jax::build()` once a typed-vs-dynamic registration API is added.
