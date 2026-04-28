# Wasm Shard Example

This example shows the end-to-end shape for a Rust-authored wasm shard mounted
alongside a host-side typed shard.

## Layout

```text
guest-shard/
  A Rust crate compiled to a wasm component.

typed-shard/
  A Rust crate that defines a host-side typed shard.

host-app/
  A Rust application that registers the typed shard, starts Jax, loads the wasm shard, and mounts it at runtime.
```

## Intended Flow

1. The guest shard declares stable metadata through a manifest.
2. The guest shard exports descriptor, JSON config schema, configuration, and lifecycle functions described by `jax-shard.wit`.
3. The application builds and starts Jax with the typed shard registered.
4. The host loader reads the wasm file, validates the manifest, and creates a reusable wasm shard module.
5. The application reads the module config schema and instantiates that module, optionally passing JSON instance configuration.
6. The application mounts the resulting `Arc<dyn jax::Shard>` into the running Jax instance.
7. The application can unmount the wasm shard before shutting down the typed shard.

## Boundary

`jax` core does not know about wasm. The host adapter is responsible for turning
a wasm component into `Arc<dyn jax::Shard>`.
