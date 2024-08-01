use std::{
  sync::atomic::Ordering,
  sync::{Arc, Mutex, Condvar},
  ops::DerefMut,
  env::var,
  string::ToString,
};
use nng::{Aio, AioResult, Context, Protocol, Socket};

use crate::log_error;
//use crate::log_warning;
use crate::log_info;
//use crate::log_verbose;
use crate::log_trace;
use crate::error;
use crate::log::*;
use crate::protocol::*;
use crate::operations::*;

pub use meritrank::Weight;

lazy_static::lazy_static! {
  pub static ref THREADS : usize =
    var("MERITRANK_SERVICE_THREADS")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(1);

  static ref SERVICE_URL : String =
    var("MERITRANK_SERVICE_URL")
      .unwrap_or("tcp://127.0.0.1:10234".to_string());
}

#[derive(Clone)]
pub struct Data {
  pub graph_readable : Arc<Mutex<AugMultiGraph>>,
  pub graph_writable : Arc<Mutex<AugMultiGraph>>,
  pub queue_commands : Arc<Mutex<Vec<Command>>>,
  pub cond_add       : Arc<Condvar>,
  pub cond_done      : Arc<Condvar>,
}

fn perform_command(
  data    : &Data,
  command : Command
) -> Result<Vec<u8>, BoxedError> {
  log_trace!("perform_command");

  if command.id == CMD_RESET            ||
     command.id == CMD_RECALCULATE_ZERO ||
     command.id == CMD_DELETE_EDGE      ||
     command.id == CMD_DELETE_NODE      ||
     command.id == CMD_PUT_EDGE
  {
    //  Write commands

    let mut res : Option<_> = None;
    let mut graph = match data.graph_writable.lock() {
      Ok(x)  => x,
      Err(e) => return error!("perform_command", "{}", e),
    };

    match command.id.as_str() {
      CMD_RESET => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          res = Some(graph.write_reset());
        }
      },
      CMD_RECALCULATE_ZERO => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          res = Some(graph.write_recalculate_zero());
        }
      },
      CMD_DELETE_EDGE => {
        if let Ok((src, dst)) = rmp_serde::from_slice(command.payload.as_slice()) {
          res = Some(graph.write_delete_edge(command.context.as_str(), src, dst));
        }
      },
      CMD_DELETE_NODE => {
        if let Ok(node) = rmp_serde::from_slice(command.payload.as_slice()) {
          res = Some(graph.write_delete_node(command.context.as_str(), node));
        }
      },
      CMD_PUT_EDGE => {
        if let Ok((src, dst, amount)) = rmp_serde::from_slice(command.payload.as_slice()) {
          res = Some(graph.write_put_edge(command.context.as_str(), src, dst, amount));
        }
      },
      _ => {},
    };
    match data.graph_readable.lock() {
      Ok(ref mut x) => {
        x.copy_from(graph.deref_mut());
      }
      Err(e) => {
        return error!("perform_command", "{}", e);
      },
    };

    if let Some(x) = res {
      return x;
    }
  } else if command.id == CMD_SYNC {
    if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
      let mut queue = data.queue_commands.lock().expect("Mutex lock failed");
      while !queue.is_empty() {
        log_trace!("wait for queue to be empty");
        queue = data.cond_done.wait(queue).expect("Condvar wait failed");
      }
      return Ok(rmp_serde::to_vec(&())?);
    }
  } else if command.id == CMD_VERSION {
    if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
      return read_version();
    }
  } else if command.id == CMD_LOG_LEVEL {
    if let Ok(log_level) = rmp_serde::from_slice(command.payload.as_slice()) {
      return write_log_level(log_level);
    }
  } else {
    //  Read commands

    let mut graph = match data.graph_readable.lock() {
      Ok(x)  => x,
      Err(e) => {
        return error!("perform_command", "{}", e);
      },
    };
    match command.id.as_str() {
      CMD_NODE_LIST => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_node_list();
        }
      },
      CMD_NODE_SCORE => {
        if let Ok((ego, target)) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_node_score(command.context.as_str(), ego, target);
        }
      },
      CMD_SCORES => {
        if let Ok((ego, kind, hide_personal, lt, lte, gt, gte, index, count)) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_scores(command.context.as_str(), ego, kind, hide_personal, lt, lte, gt, gte, index, count);
        }
      },
      CMD_GRAPH => {
        if let Ok((ego, focus, positive_only, index, count)) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_graph(command.context.as_str(), ego, focus, positive_only, index, count);
        }
      },
      CMD_CONNECTED => {
        if let Ok(node) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_connected(command.context.as_str(), node);
        }
      },
      CMD_EDGES => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_edges(command.context.as_str());
        }
      },
      CMD_MUTUAL_SCORES => {
        if let Ok(ego) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_mutual_scores(command.context.as_str(), ego);
        }
      },
      _ => {
        return error!("perform_command", "Unknown command: `{}`", command.id);
      }
    }
  }

  return error!("perform_command", "Invalid payload for command `{}`: {:?}", command.id.as_str(), command.payload);
}

fn command_queue_thread(data : Data) {
  let mut queue = data.queue_commands.lock().expect("Mutex lock failed");
  log_trace!("command_queue_thread");

  loop {
    let commands : Vec<_> = queue.clone();
    queue.clear();
    std::mem::drop(queue);

    for cmd in commands {
      match perform_command(&data, cmd.clone()) {
        Ok(_)  => {},
        Err(e) => {
          log_error!("(write_queue_thread) {}", e);
        },
      };
    }

    queue = data.queue_commands.lock().expect("Mutex lock failed");
    log_trace!("notify done");
    data.cond_done.notify_all();
    queue = data.cond_add.wait(queue).expect("Condvar wait failed");
  }
}

fn put_for_write(
  data    : &Data,
  command : Command,
) {
  log_trace!("put_for_write");

  let mut queue = data.queue_commands.lock().expect("Mutex lock failed");
  queue.push(command);
  log_trace!("notify add");
  data.cond_add.notify_one();
}

fn decode_and_handle_request(
  data    : Data,
  request : &[u8]
) -> Result<Vec<u8>, BoxedError> {
  log_trace!("decode_and_handle_request");

  let command : Command;

  match rmp_serde::from_slice(request) {
    Ok((command_value, context_value, blocking_value, payload_value)) => {
      command = Command {
        id       : command_value,
        context  : context_value,
        blocking : blocking_value,
        payload  : payload_value,
      };

      if command.context.is_empty() {
        log_trace!("decoded command `{}` in NULL with payload {:?}", command.id, command.payload);
      } else {
        log_trace!("decoded command `{}` in `{}` with payload {:?}", command.id, command.context, command.payload);
      }
    },

    Err(error) =>
      return error!("decode_and_handle_request", "Invalid request: {:?}; decoding error: {}", request, error),
  }

  if !command.context.is_empty() && (
    command.id == CMD_VERSION          ||
    command.id == CMD_LOG_LEVEL        ||
    command.id == CMD_RESET            ||
    command.id == CMD_RECALCULATE_ZERO ||
    command.id == CMD_NODE_LIST
  ) {
    return error!("decode_and_handle_request", "Context should be empty");
  }

  if !command.blocking {
    put_for_write(&data, command);
    Ok(rmp_serde::to_vec(&())?)
  } else {
    perform_command(&data, command)
  }
}

fn worker_callback(
  data : Data,
  aio  : Aio,
  ctx  : &Context,
  res  : AioResult
) {
  log_trace!("worker_callback");

  match res {
    AioResult::Send(Ok(_)) => {
      match ctx.recv(&aio) {
        Ok(_) => {},
        Err(error) => {
          log_error!("(worker_callback) RECV failed: {}", error);
        },
      }
    },

    AioResult::Recv(Ok(req)) => {
      let msg : Vec<u8> = match decode_and_handle_request(data, req.as_slice()) {
        Ok(bytes)  => bytes,
        Err(error) => match rmp_serde::to_vec(&error.to_string()) {
          Ok(bytes)  => bytes,
          Err(error) => {
            log_error!("(worker_callback) Unable to serialize error: {:?}", error);
            vec![]
          },
        },
      };
      match ctx.send(&aio, msg.as_slice()) {
        Ok(_) => {},
        Err(error) => {
          log_error!("(worker_callback) SEND failed: {:?}", error);
        }
      };
    }

    AioResult::Sleep(_) => {},

    AioResult::Send(Err(error)) => {
      log_error!("(worker_callback) Async SEND failed: {:?}", error);
    },

    AioResult::Recv(Err(error)) => {
      log_error!("(worker_callback) Async RECV failed: {:?}", error);
    },
  };
}

pub fn main_async(threads : usize) -> Result<(), BoxedError> {
  let threads = if threads < 1 { 1 } else { threads };

  log_info!("Starting server {} at {}, {} threads", VERSION, *SERVICE_URL, threads);
  log_info!("NUM_WALK={}", *NUM_WALK);

  let data = Data {
    graph_readable : Arc::<Mutex<AugMultiGraph>>::new(Mutex::<AugMultiGraph>::new(AugMultiGraph::new()?)),
    graph_writable : Arc::<Mutex<AugMultiGraph>>::new(Mutex::<AugMultiGraph>::new(AugMultiGraph::new()?)),
    queue_commands : Arc::<Mutex<Vec<Command>>>::new(Mutex::<Vec<Command>>::new(vec![])),
    cond_add       : Arc::<Condvar>::new(Condvar::new()),
    cond_done      : Arc::<Condvar>::new(Condvar::new()),
  };

  let data_cloned = data.clone();

  std::thread::spawn(move || {
    command_queue_thread(data_cloned);
  });

  let s = Socket::new(Protocol::Rep0)?;

  let workers : Vec<_> = (0..threads)
    .map(|_| {
      let ctx         = Context::new(&s)?;
      let ctx_cloned  = ctx.clone();
      let data_cloned = data.clone();

      let aio = Aio::new(move |aio, res| {
        worker_callback(
          data_cloned.clone(),
          aio,
          &ctx_cloned,
          res
        );
      })?;

      Ok((aio, ctx))
    })
    .collect::<Result<_, nng::Error>>()?;

  s.listen(&SERVICE_URL)?;

  for (a, c) in &workers {
    c.recv(a)?;
  }

  std::thread::park();
  Ok(())
}
