extern crate alloc;
extern crate jax as jax_core;

use alloc::sync::Arc;
use core::error::Error;
use jax_core::{Dependency, Descriptor, Jax, JaxResult, Shard, ShardId};
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

wasmtime::component::bindgen!({
    world: "jax-shard",
    path: "../jax-wasm-abi/wit",
});

pub struct WasmShardLoader {
    engine: Engine,
}

pub struct WasmShardModule {
    engine: Engine,
    component: Component,
}

impl WasmShardLoader {
    pub fn new() -> JaxResult<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config)?;
        Ok(Self { engine })
    }

    pub async fn load_from_file(&self, path: impl AsRef<Path>) -> JaxResult<WasmShardModule> {
        let bytes = std::fs::read(path)?;
        self.load_from_bytes(&bytes).await
    }

    pub async fn load_from_bytes(&self, bytes: &[u8]) -> JaxResult<WasmShardModule> {
        let component = Component::from_binary(&self.engine, bytes)?;
        Ok(WasmShardModule {
            engine: self.engine.clone(),
            component,
        })
    }
}

impl WasmShardModule {
    pub fn config_schema_json(&self) -> JaxResult<String> {
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        JaxShard::add_to_linker::<_, HasSelf<HostState>>(&mut linker, |state| state)?;

        let mut store = Store::new(&self.engine, HostState::new());
        let bindings = JaxShard::instantiate(&mut store, &self.component, &linker)?;
        Ok(bindings.interface0.call_config_schema(&mut store)?)
    }

    pub fn config_schema(&self) -> JaxResult<serde_json::Value> {
        Ok(serde_json::from_str(&self.config_schema_json()?)?)
    }

    pub fn instantiate(&self) -> JaxResult<Arc<dyn Shard>> {
        self.instantiate_with_config(&())
    }

    pub fn instantiate_with_config<T: Serialize + ?Sized>(
        &self,
        config: &T,
    ) -> JaxResult<Arc<dyn Shard>> {
        let config = serde_json::to_string(config)?;
        self.instantiate_with_json(config)
    }

    pub fn instantiate_with_json(&self, config: impl Into<String>) -> JaxResult<Arc<dyn Shard>> {
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        JaxShard::add_to_linker::<_, HasSelf<HostState>>(&mut linker, |state| state)?;

        let mut store = Store::new(&self.engine, HostState::new());
        let bindings = JaxShard::instantiate(&mut store, &self.component, &linker)?;
        let descriptor =
            wit_descriptor_to_runtime_descriptor(bindings.interface0.call_describe(&mut store)?)?;
        let config = config.into();
        guest_unit_result_to_jax(bindings.interface0.call_configure(&mut store, &config)?)?;

        Ok(Arc::new(ComponentShard {
            descriptor,
            instance: Mutex::new(ComponentInstance { store, bindings }),
        }))
    }
}

impl Default for WasmShardLoader {
    fn default() -> Self {
        Self::new().unwrap_or_else(|error| {
            panic!("Jax WASM host: failed to create loader: {error}");
        })
    }
}

struct ComponentShard {
    descriptor: Descriptor,
    instance: Mutex<ComponentInstance>,
}

#[async_trait::async_trait]
impl Shard for ComponentShard {
    fn descriptor(&self) -> Descriptor {
        self.descriptor.clone()
    }

    async fn setup(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        self.with_instance(|bindings, store| bindings.interface0.call_setup(store))
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        self.with_instance(|bindings, store| bindings.interface0.call_teardown(store))
    }
}

impl ComponentShard {
    fn with_instance(
        &self,
        call: impl FnOnce(&JaxShard, &mut Store<HostState>) -> wasmtime::Result<Result<(), String>>,
    ) -> JaxResult<()> {
        let mut guard = self
            .instance
            .lock()
            .map_err(|_| "Jax WASM host: component instance lock poisoned")?;
        let ComponentInstance { store, bindings } = &mut *guard;
        guest_unit_result_to_jax(call(bindings, store)?)
    }
}

struct ComponentInstance {
    store: Store<HostState>,
    bindings: JaxShard,
}

struct HostState {
    wasi: WasiCtx,
    table: ResourceTable,
}

impl HostState {
    fn new() -> Self {
        Self {
            wasi: WasiCtx::builder().build(),
            table: ResourceTable::new(),
        }
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl jax::wasm::host::Host for HostState {
    fn log(&mut self, level: String, message: String) {
        println!("[wasm:{level}] {message}");
    }
}

fn wit_descriptor_to_runtime_descriptor(
    descriptor: exports::jax::wasm::shard::Descriptor,
) -> JaxResult<Descriptor> {
    let id: ShardId = descriptor.id.parse()?;
    let dependencies = descriptor
        .dependencies
        .into_iter()
        .map(|dependency| {
            dependency
                .id
                .parse::<ShardId>()
                .map(Dependency::new)
                .map_err(|error| Box::new(error) as Box<dyn Error + Send + Sync>)
        })
        .collect::<Result<_, _>>()?;

    Ok(Descriptor::new(id)
        .with_label(descriptor.label)
        .with_dependencies(dependencies))
}

fn guest_unit_result_to_jax(result: Result<(), String>) -> JaxResult<()> {
    match result {
        Ok(()) => Ok(()),
        Err(message) => Err(message.into()),
    }
}
