# Wasm Shard Example

This example shows the end-to-end shape for a Rust-authored wasm shard mounted
alongside a host-side typed shard.

## Layout

```text
guest-shard/
  A Rust crate compiled to a wasm component.

typed-shard/
  A Rust crate that defines a host-side typed shard and exports a wasm-visible capability.

host-app/
  A Rust application that registers the typed shard, starts Jax, loads the wasm shard, and mounts it at runtime.
```

## Intended Flow

1. The guest shard declares stable metadata through a manifest.
2. The guest shard exports descriptor, JSON config schema, configuration, and lifecycle functions described by `jax-shard.wit`.
3. The typed shard marks wasm-visible methods with `#[jax_wasm::export]`.
4. The guest imports that capability with `jax_wasm::import!(example_wasm_typed_shard::LogShard)`.
5. The application builds and starts Jax with the typed shard registered.
6. The host loader reads the wasm file and creates a reusable wasm shard module.
7. The application reads the module config schema and instantiates that module with host exports.
8. The application mounts the resulting `Arc<dyn jax::Shard>` into the running Jax instance.
9. The wasm shard can call the exported typed shard capability during setup and teardown.
10. The application can unmount the wasm shard before shutting down the typed shard.

## Boundary

`jax` core does not know about wasm. The host adapter is responsible for turning
a wasm component into `Arc<dyn jax::Shard>` and for wiring exported typed shard
capabilities into the wasm runtime.
