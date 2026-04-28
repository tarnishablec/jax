extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde_json::Value;
use tracing::Level;

pub use jax_wasm_macros::{export, import};

pub trait WasmExport: Send + Sync + 'static {
    fn shard_id(&self) -> String;

    fn dispatch(&self, method: &str, request_json: &str) -> Result<String, String>;
}

pub trait ToWireValue {
    fn to_wire_value(&self) -> Result<Value, String>;
}

pub trait FromWireValue: Sized {
    fn from_wire_value(value: Value) -> Result<Self, String>;
}

pub trait IntoWireResult {
    fn into_wire_result(self) -> Result<String, String>;
}

impl ToWireValue for String {
    fn to_wire_value(&self) -> Result<Value, String> {
        Ok(Value::String(self.clone()))
    }
}

impl ToWireValue for Level {
    fn to_wire_value(&self) -> Result<Value, String> {
        Ok(Value::String(self.to_string()))
    }
}

impl FromWireValue for String {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        value
            .as_str()
            .map(ToString::to_string)
            .ok_or_else(|| "Jax WASM: expected string wire value".to_string())
    }
}

impl FromWireValue for Level {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        let input = String::from_wire_value(value)?;
        input
            .parse()
            .map_err(|error| format!("Jax WASM: invalid tracing level [{input}]: {error}"))
    }
}

impl IntoWireResult for () {
    fn into_wire_result(self) -> Result<String, String> {
        Ok("null".into())
    }
}

impl<T, E> IntoWireResult for Result<T, E>
where
    T: IntoWireResult,
    E: core::fmt::Display,
{
    fn into_wire_result(self) -> Result<String, String> {
        self.map_err(|error| error.to_string())?.into_wire_result()
    }
}

pub fn encode_request(values: Vec<Value>) -> Result<String, String> {
    serde_json::to_string(&values).map_err(|error| error.to_string())
}

pub fn decode_request(request_json: &str) -> Result<Vec<Value>, String> {
    serde_json::from_str(request_json).map_err(|error| error.to_string())
}

pub fn decode_response<T: FromWireValue>(response_json: &str) -> Result<T, String> {
    let value = serde_json::from_str(response_json).map_err(|error| error.to_string())?;
    T::from_wire_value(value)
}

impl FromWireValue for () {
    fn from_wire_value(value: Value) -> Result<Self, String> {
        if value.is_null() {
            Ok(())
        } else {
            Err("Jax WASM: expected null wire response".into())
        }
    }
}
