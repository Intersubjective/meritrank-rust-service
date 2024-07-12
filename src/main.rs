pub mod service;
pub mod commands;
pub mod astar;

#[cfg(test)]
mod tests;

use crate::service::{main_async, main_sync, THREADS};
use ctrlc;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
  ctrlc::set_handler(move || {
    println!("");
    std::process::exit(0)
  })?;

  if *THREADS > 1 {
    main_async(*THREADS)
  } else {
    main_sync()
  }
} 
