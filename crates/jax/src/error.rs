use alloc::boxed::Box;
use core::error::Error;

pub type JaxError = Box<dyn Error + Send + Sync>;
pub type JaxResult<T> = Result<T, JaxError>;
