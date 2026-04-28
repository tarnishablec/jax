extern crate alloc;
extern crate jax as jax_core;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::error::Error;
use jax_core::{Dependency, Descriptor, Jax, JaxResult, Shard, ShardId};
use jax_wasm::WasmExport;
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

mod jax_wit {
    wasmtime::component::bindgen!({
        world: "jax-shard",
        path: "../jax-wasm-abi/wit",
    });

    pub mod runtime {
        pub use super::jax::wit::runtime::*;
    }

    pub mod shard {
        pub use super::exports::jax::wit::shard::*;
    }

    pub fn call_config_schema(
        bindings: &JaxShard,
        store: &mut ::wasmtime::Store<super::HostState>,
    ) -> ::wasmtime::Result<String> {
        bindings.interface0.call_config_schema(store)
    }

    pub fn call_describe(
        bindings: &JaxShard,
        store: &mut ::wasmtime::Store<super::HostState>,
    ) -> ::wasmtime::Result<shard::Descriptor> {
        bindings.interface0.call_describe(store)
    }

    pub fn call_configure(
        bindings: &JaxShard,
        store: &mut ::wasmtime::Store<super::HostState>,
        config: &str,
    ) -> ::wasmtime::Result<Result<(), String>> {
        bindings.interface0.call_configure(store, config)
    }

    pub fn call_setup(
        bindings: &JaxShard,
        store: &mut ::wasmtime::Store<super::HostState>,
    ) -> ::wasmtime::Result<Result<(), String>> {
        bindings.interface0.call_setup(store)
    }

    pub fn call_teardown(
        bindings: &JaxShard,
        store: &mut ::wasmtime::Store<super::HostState>,
    ) -> ::wasmtime::Result<Result<(), String>> {
        bindings.interface0.call_teardown(store)
    }
}

pub struct WasmShardLoader {
    engine: Engine,
}

pub struct WasmShardModule {
    engine: Engine,
    component: Component,
}

#[derive(Default)]
pub struct WasmHostExports {
    exports: BTreeMap<String, Arc<dyn WasmExport>>,
}

impl WasmHostExports {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_export<T>(mut self, export: Arc<T>) -> Self
    where
        T: WasmExport,
    {
        self.insert(export);
        self
    }

    pub fn insert<T>(&mut self, export: Arc<T>)
    where
        T: WasmExport,
    {
        self.exports.insert(export.shard_id(), export);
    }

    fn dispatch(&self, shard_id: &str, method: &str, request_json: &str) -> Result<String, String> {
        let export = self
            .exports
            .get(shard_id)
            .ok_or_else(|| alloc::format!("Jax WASM: shard [{}] is not exported", shard_id))?;
        export.dispatch(method, request_json)
    }
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
        jax_wit::JaxShard::add_to_linker::<_, HasSelf<HostState>>(&mut linker, |state| state)?;

        let mut store = Store::new(&self.engine, HostState::new());
        let bindings = jax_wit::JaxShard::instantiate(&mut store, &self.component, &linker)?;
        Ok(jax_wit::call_config_schema(&bindings, &mut store)?)
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
        self.instantiate_with_json_on(None, Arc::new(WasmHostExports::new()), config)
    }

    pub fn instantiate_with_config_on<T: Serialize + ?Sized>(
        &self,
        jax: Arc<Jax>,
        exports: Arc<WasmHostExports>,
        config: &T,
    ) -> JaxResult<Arc<dyn Shard>> {
        let config = serde_json::to_string(config)?;
        self.instantiate_with_json_on(Some(jax), exports, config)
    }

    pub fn instantiate_with_json_on(
        &self,
        jax: Option<Arc<Jax>>,
        exports: Arc<WasmHostExports>,
        config: impl Into<String>,
    ) -> JaxResult<Arc<dyn Shard>> {
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        jax_wit::JaxShard::add_to_linker::<_, HasSelf<HostState>>(&mut linker, |state| state)?;

        let mut store = Store::new(&self.engine, HostState::with_runtime(jax, exports));
        let bindings = jax_wit::JaxShard::instantiate(&mut store, &self.component, &linker)?;
        let descriptor =
            wit_descriptor_to_runtime_descriptor(jax_wit::call_describe(&bindings, &mut store)?)?;
        let config = config.into();
        guest_unit_result_to_jax(jax_wit::call_configure(&bindings, &mut store, &config)?)?;

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
        self.with_instance(jax_wit::call_setup)
    }

    async fn teardown(&self, _jax: Arc<Jax>) -> JaxResult<()> {
        self.with_instance(jax_wit::call_teardown)
    }
}

impl ComponentShard {
    fn with_instance(
        &self,
        call: impl FnOnce(
            &jax_wit::JaxShard,
            &mut Store<HostState>,
        ) -> wasmtime::Result<Result<(), String>>,
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
    bindings: jax_wit::JaxShard,
}

struct HostState {
    wasi: WasiCtx,
    table: ResourceTable,
    jax: Option<Arc<Jax>>,
    exports: Arc<WasmHostExports>,
}

impl HostState {
    fn new() -> Self {
        Self::with_runtime(None, Arc::new(WasmHostExports::new()))
    }

    fn with_runtime(jax: Option<Arc<Jax>>, exports: Arc<WasmHostExports>) -> Self {
        Self {
            wasi: WasiCtx::builder().build(),
            table: ResourceTable::new(),
            jax,
            exports,
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

impl jax_wit::runtime::Host for HostState {
    fn call_shard(
        &mut self,
        shard_id: String,
        method: String,
        request_json: String,
    ) -> Result<String, String> {
        let _ = self
            .jax
            .as_ref()
            .ok_or_else(|| "Jax WASM: runtime is not attached".to_string())?;
        self.exports.dispatch(&shard_id, &method, &request_json)
    }
}

fn wit_descriptor_to_runtime_descriptor(
    descriptor: jax_wit::shard::Descriptor,
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
