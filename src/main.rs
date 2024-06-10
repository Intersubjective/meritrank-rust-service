//  FIXME
//  Floating-point arithmetic for weight calculation will
//  break invariance.
//

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use meritrank::{MeritRank, MyGraph};
use std::thread;
use std::time::Duration;
use std::env::var;
use std::string::ToString;
use std::sync::MutexGuard;
use std::error::Error;
use petgraph::graph::{EdgeIndex, NodeIndex};
use nng::{Aio, AioResult, Context, Message, Protocol, Socket};
use simple_pagerank::Pagerank;
use meritrank::{MeritRankError, Weight, NodeId};
use ctrlc;

//  ================================================
//
//    ...Previously called mrgraph
//
//  ================================================

lazy_static::lazy_static! {
  pub static ref GRAPH: Arc<Mutex<GraphSingleton>> = Arc::new(Mutex::new(GraphSingleton::new()));
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum NodeType {
  NodeUnknown,
  NodeUser,
  NodeBeacon,
  NodeComment,
}

#[derive(Clone, Copy)]
pub struct NodeInfo {
  pub id   : NodeId,
  pub kind : NodeType,
}

pub struct NodesInfo {
  //  FIXME(Performance)
  //  Use parallel array of node names for O(0) lookup.
  //  Add node type attributes.

  node_names : HashMap<String, NodeInfo>,
  node_count : u64,
}

impl NodesInfo {
  pub fn new() -> NodesInfo {
    NodesInfo {
      node_names : HashMap::new(),
      node_count : 0,
    }
  }

  pub fn node_name_to_id_locked(&self, node_name: &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    Ok(self.node_names.get(node_name).unwrap().id)
  }

  pub fn node_id_to_name_locked(&self, node_id: NodeId) -> Result<String, Box<dyn Error + 'static>> {
    for (name, n) in self.node_names.iter() {
      if n.id == node_id {
        return Ok(name.to_string());
      }
    }
    Err("Node not found".into())
  }
}

pub struct GraphSingleton {
  pub graph  : MeritRank,                  // null-context
  pub graphs : HashMap<String, MeritRank>, // contexted
  pub info   : NodesInfo,
}

impl GraphSingleton {
  /// Constructor
  pub fn new() -> GraphSingleton {
    GraphSingleton {
      graph  : MeritRank::new(MyGraph::new()).unwrap(),
      graphs : HashMap::new(),
      info   : NodesInfo::new(),
    }
  }

  pub fn reset(&mut self) {
    self.graph  = MeritRank::new(MyGraph::new()).unwrap();
    self.graphs = HashMap::new();
    self.info   = NodesInfo::new();
  }

  pub fn contexts() -> Result<Vec<String>, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.graphs.keys().map(|ctx| ctx.clone()).collect_vec())
  }

  pub fn add_node(node_name : &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.add_node_id(node_name))
  }

  pub fn add_node_id(&mut self, node_name : &str) -> NodeId {
    if let Some(&n) = self.info.node_names.get(node_name) {
      n.id
    } else {
      let node_id = self.info.node_count;
      self.info.node_count += 1;
      self.info.node_names.insert(node_name.to_string(), NodeInfo {
        id   : node_id,
        kind : match node_name.chars().nth(0).unwrap() {
          'U' => NodeType::NodeUser,
          'B' => NodeType::NodeBeacon,
          'C' => NodeType::NodeComment,
          _   => NodeType::NodeUnknown,
        },
      });
      self.graph.add_node(node_id.into());
      node_id
    }
  }

  pub fn add_node_id_contexted(&mut self, context: &str, node_name: &str) -> NodeId {
    if let Some(&n) = self.info.node_names.get(node_name) {
      n.id
    } else {
      let node_id = self.add_node_id(node_name); // create a node in null-context
      if !self.graphs.contains_key(context) {
        self.graphs.insert(context.to_string(), MeritRank::new(MyGraph::new()).unwrap());
      }
      let graph = self.graphs.get_mut(context).unwrap();
      graph.add_node(node_id.into());
      node_id
    }
  }

  pub fn node_name_to_id(node_name: &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.info.node_name_to_id_locked(node_name)?)
  }

  pub fn node_id_to_name(node_id: NodeId) -> Result<String, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.info.node_id_to_name_locked(node_id)?)
  }
}

//  ================================================
//
//  The service
//
//  ================================================

lazy_static::lazy_static! {
  static ref SERVICE_URL: String =
    var("MERITRANK_SERVICE_URL")
      .unwrap_or("tcp://127.0.0.1:10234".to_string());

  static ref THREADS : usize =
    var("MERITRANK_SERVICE_THREADS")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(1);

  static ref NUM_WALK: usize =
    var("MERITRANK_NUM_WALK")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(10000);

  static ref ZERO_NODE: String =
    var("MERITRANK_ZERO_NODE")
      .unwrap_or("U000000000000".to_string());

  static ref TOP_NODES_LIMIT: usize =
    var("MERITRANK_TOP_NODES_LIMIT")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(100);

  static ref EMPTY_RESULT: Vec<u8> = {
    const EMPTY_ROWS_VEC: Vec<(&str, &str, f64)> = Vec::new();
    rmp_serde::to_vec(&EMPTY_ROWS_VEC).unwrap()
  };
}

const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

fn main() -> Result<(), Box<dyn Error + 'static>> {
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

fn main_sync() -> Result<(), Box<dyn Error + 'static>> {
  println!("Starting server {} at {}", VERSION.unwrap_or("unknown"), *SERVICE_URL);
  println!("NUM_WALK={}", *NUM_WALK);

  let s = Socket::new(Protocol::Rep0)?;
  s.listen(&SERVICE_URL)?;

  loop {
    let request: Message = s.recv()?;
    let reply: Vec<u8> = process(request);
    let _ = s.send(reply.as_slice()).map_err(|(_, e)| e)?;
  }
  // Ok(())
}

fn main_async(threads : usize) -> Result<(), Box<dyn Error + 'static>> {
  println!("Starting server {} at {}, {} threads", VERSION.unwrap_or("unknown"), *SERVICE_URL, threads);
  println!("NUM_WALK={}", *NUM_WALK);

  let s = Socket::new(Protocol::Rep0)?;

  // Create all of the worker contexts
  let workers: Vec<_> = (0..threads)
    .map(|_| {
      let ctx = Context::new(&s)?;
      let ctx_clone = ctx.clone();
      let aio = Aio::new(move |aio, res| worker_callback(aio, &ctx_clone, res))?;
      Ok((aio, ctx))
    })
    .collect::<Result<_, nng::Error>>()?;

  // Only after we have the workers do we start listening.
  s.listen(&SERVICE_URL)?;

  // Now start all of the workers listening.
  for (a, c) in &workers {
    c.recv(a)?;
  }

  thread::sleep(Duration::from_secs(60 * 60 * 24 * 365)); // 1 year

  Ok(())
}

/// Callback function for workers.
fn worker_callback(aio: Aio, ctx: &Context, res: AioResult) {
  match res {
    // We successfully sent the message, wait for a new one.
    AioResult::Send(Ok(_)) => ctx.recv(&aio).unwrap(),

    // We successfully received a message.
    AioResult::Recv(Ok(req)) => {
      let msg: Vec<u8> = process(req);
      ctx.send(&aio, msg.as_slice()).unwrap();
    }

    AioResult::Sleep(_) => {},

    // Anything else is an error and we will just panic.
    AioResult::Send(Err(e)) =>
      panic!("Error: {}", e.1),

    AioResult::Recv(Err(e)) =>
      panic!("Error: {}", e)
  }
}

fn process(req: Message) -> Vec<u8> {
  let slice = req.as_slice();

  let ctx = GraphContext::null();

  ctx.process(slice)
    .map(|msg| msg)
    .unwrap_or_else(|e| {
      let s: String = e.to_string();
      rmp_serde::to_vec(&s).unwrap()
    })
}

fn mr_service() -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let s: String = VERSION.unwrap_or("unknown").to_string();
  Ok(rmp_serde::to_vec(&s)?)
}

fn mr_node_score_null(ego : &str, target : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let mut graph = GRAPH.lock()?;

  let ego_id    = graph.info.node_name_to_id_locked(ego).unwrap();
  let target_id = graph.info.node_name_to_id_locked(target).unwrap();

  let mut w : Weight = 0.0;
  for (_, rank) in graph.graphs.iter_mut() {
    w += match rank.get_node_score(ego_id, target_id) {
      Err(MeritRankError::NodeIsNotCalculated) => {
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        rank.get_node_score(ego_id, target_id)?
      },
      Err(x) => return Err(x.into()),
      Ok(score) => score,
    }
  }

  let result : Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
  Ok(rmp_serde::to_vec(&result)?)
}

fn mr_scores_null(ego : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let mut graph = GRAPH.lock()?;

  let ego_id = graph.info.node_name_to_id_locked(ego)?;

  let result: Vec<_> =
    graph.graphs
      .iter_mut()
      .filter_map(|(_context, rank)| {
        let rank_result = match rank.get_ranks(ego_id, None) {
          Err(MeritRankError::NodeIsNotCalculated) => {
            let _ = rank.calculate(ego_id, *NUM_WALK).ok()?;
            rank.get_ranks(ego_id, None).ok()
          },
          other => other.ok()
        };
        let rows: Vec<_> =
          rank_result?
            .into_iter()
            .map(|(n, s)| {
              (
                (
                  ego,
                  GraphSingleton::node_id_to_name(n)
                    .unwrap_or(n.to_string())
                ),
                s,
              )
            })
            .collect();
        Some(rows)
      })
      .flatten()
      .into_iter()
      .group_by(|(nodes, _)| nodes.clone())
      .into_iter()
      .map(|((src, target), rows)|
        (src, target, rows.map(|(_, score)| score).sum::<Weight>())
      )
      .collect();

  let v: Vec<u8> = rmp_serde::to_vec(&result)?;
  Ok(v)
}

pub struct GraphContext {
  context : String,
}

impl GraphContext {
  pub fn null() -> GraphContext {
    GraphContext {
      context : String::new(),
    }
  }
  pub fn new(context_init: &str) -> GraphContext {
    if context_init.is_empty() {
      GraphContext {
        context : String::new(),
      }
    } else {
      GraphContext {
        context: context_init.to_string(),
      }
    }
  }

  pub fn process_context(context: &str, payload: Vec<u8>)  -> Result<Vec<u8>, Box<dyn Error>> {
    GraphContext::new(&context).process(payload.as_slice())
  }

  pub fn process(&self, slice: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if let Ok("ver") = rmp_serde::from_slice(slice) {
      mr_service()
    } else if let Ok(((("src", "=", ego), ("dest", "=", target)), (), "null")) = rmp_serde::from_slice(slice) {
      mr_node_score_null(ego, target)
    } else if let Ok(((("src", "=", ego), ), (), "null")) = rmp_serde::from_slice(slice) {
      mr_scores_null(ego)
    } else if let Ok(("context", context, payload)) = rmp_serde::from_slice(slice) { // rmp_serde::from_slice::<(&str, &str, Vec<u8>)>(slice) {
      Self::process_context(context, payload)
    } else if let Ok(((("src", "=", ego), ("dest", "=", target)), ())) = rmp_serde::from_slice(slice) {
      self.mr_node_score(ego, target)
    } else if let Ok(((("src", "=", ego), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, "", false, f64::MIN, true, f64::MAX, true, None)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, false, score_gt, false, score_lt, false, None)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">=", score_gte), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, false, score_gte, true, score_lt, false, None)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, false, score_gt, false, score_lt, true, None)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">=", score_gte), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, false, score_gte, true, score_lt, true, None)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, hide_personal, score_gt, false, score_lt, false, limit)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">=", score_gte), ("score", "<", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, hide_personal, score_gte, true, score_lt, false, limit)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<=", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, hide_personal, score_gt, false, score_lt, true, limit)
    } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">=", score_gte), ("score", "<=", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, target_like, hide_personal, score_gte, true, score_lt, true, limit)
    } else if let Ok((((subject, object, amount), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_put_edge(subject, object, amount)
    } else if let Ok(((("src", "delete", ego), ("dest", "delete", target)), ())) = rmp_serde::from_slice(slice) {
      self.mr_delete_edge(ego, target)
    } else if let Ok(((("src", "delete", ego), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_delete_node(ego)
    } else if let Ok((((ego, "gravity", focus), positive_only, limit), ())) = rmp_serde::from_slice(slice) {
      self.mr_gravity_graph(ego, focus, positive_only/* true */, limit/* 3 */)
    } else if let Ok((((ego, "gravity_nodes", focus), positive_only, limit), ())) = rmp_serde::from_slice(slice) {
      self.mr_gravity_nodes(ego, focus, positive_only /* false */, limit /* 3 */)
    } else if let Ok((((ego, "connected"), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_connected(ego)
    } else if let Ok(("nodes", ())) = rmp_serde::from_slice(slice) {
      self.mr_nodes()
    } else if let Ok(("edges", ())) = rmp_serde::from_slice(slice) {
      self.mr_edges()
    } else if let Ok(("reset", ())) = rmp_serde::from_slice(slice) {
      self.mr_reset()
    } else if let Ok(("zerorec", ())) = rmp_serde::from_slice(slice) {
      self.mr_zerorec()
    } else {
      let err: String = format!("Error: Cannot understand request {:?}", slice);
      Err(err.into())
    }
  }

  fn mr_node_score(
    &self,
    ego    : &str,
    target : &str
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let mut graph = GRAPH.lock()?;

    let ego_id    = graph.info.node_name_to_id_locked(ego)?;
    let target_id = graph.info.node_name_to_id_locked(target)?;
    let rank      = &mut graph.graph;

    let w = match rank.get_node_score(ego_id, target_id) {
      Err(MeritRankError::NodeIsNotCalculated) => {
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        rank.get_node_score(ego_id, target_id)?
      },
      Err(x)    => return Err(x.into()),
      Ok(score) => score,
    };

    let result: Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  fn mr_scores(
    &self,
    ego           : &str,
    target_like   : &str,
    hide_personal : bool,
    score_lt      : f64,
    score_lte     : bool,
    score_gt      : f64,
    score_gte     : bool,
    limit         : Option<i32>
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>>
  {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let node_id = GraphSingleton::node_name_to_id(ego)?;

    //  NOTE
    //  We don't have to recalculate the scores.
    let _ = rank.calculate(node_id, *NUM_WALK)?;

    let ranks = rank.get_ranks(node_id, None)?;

    let intermediate = ranks
      .into_iter()
      .map(|(n, w)| {
        (
          ego,
          info.node_id_to_name_locked(n).unwrap_or(n.to_string()),
          w,
        )
      })
      .filter(|(_, target, _)| target.starts_with(target_like))
      .filter(|(_, _, score)| score_gt < *score || (score_gte && score_gt == *score))
      .filter(|(_, _, score)| *score < score_lt || (score_lte && score_lt == *score));

    let result = intermediate
      .filter(|(_ego, target, _)|
        if hide_personal {
          match info.node_name_to_id_locked(target) {
            Ok(target_id) =>
              !((target.starts_with("C") || target.starts_with("B")) &&
                rank.get_edge(target_id, node_id).is_some()),
            _ => true
          }
        } else { true }
      );

    let limited: Vec<(&str, String, Weight)> =
      match limit {
        Some(limit) => result.take(limit.try_into().unwrap()).collect(),
        None => result.collect(),
      };

    let v: Vec<u8> = rmp_serde::to_vec(&limited)?;
    Ok(v)
  }

  fn set_edge_locked(
    &self,
    graph  : &mut MutexGuard<GraphSingleton>,
    src    : NodeId,
    dst    : NodeId,
    amount : f64
  ) -> Result<(), Box<dyn Error + 'static>> {
    if self.context.is_empty() {
      graph.graph.add_edge(src, dst, amount);
    } else {
      if !graph.graphs.contains_key(self.context.as_str()) {
        graph.graphs.insert(self.context.clone(), MeritRank::new(MyGraph::new())?);
      }
      let null_weight    = graph.graph.get_edge(src, dst).unwrap_or(0.0);
      let contexted_rank = graph.graphs.get_mut(self.context.as_str()).unwrap();
      let old_weight     = contexted_rank.get_edge(src, dst).unwrap_or(0.0);

      contexted_rank.add_edge(src, dst, amount);
      graph.graph.add_edge(src, dst, null_weight + amount - old_weight);
    }

    Ok(())
  }

  fn mr_put_edge(
    &self,
    src     : &str,
    dst     : &str,
    amount  : f64
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let mut graph = GRAPH.lock()?;

    let (src_id, dst_id) =
      if self.context.is_empty() {
        ( graph.add_node_id(src),
          graph.add_node_id(dst) )
      } else {
        ( graph.add_node_id_contexted(self.context.as_str(), src),
          graph.add_node_id_contexted(self.context.as_str(), dst) )
      };

    self.set_edge_locked(&mut graph, src_id, dst_id, amount)?;

    let result: Vec<(&str, &str, f64)> = [(src, dst, amount)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  fn mr_delete_edge(
    &self,
    src : &str,
    dst : &str,
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let _ = self.mr_put_edge(src, dst, 0.0)?;
    Ok(EMPTY_RESULT.to_vec())
  }

  fn mr_delete_node(
    &self,
    ego : &str,
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let mut graph = GRAPH.lock()?;
    let     id    = graph.info.node_names.get(ego).unwrap().id;

    //  FIXME
    //  Add a function to get all neighbors in MeritRank.

    for n in graph.graph.neighbors_weighted(id, true).unwrap().keys() {
      self.set_edge_locked(&mut graph, id, *n, 0.0)?;
    }

    for n in graph.graph.neighbors_weighted(id, false).unwrap().keys() {
      self.set_edge_locked(&mut graph, id, *n, 0.0)?;
    }

    Ok(EMPTY_RESULT.to_vec())
  }

  fn gravity_graph(
    &self,
    ego           : &str,
    focus         : &str,
    positive_only : bool,
    limit         : i32
  ) -> Result<
      (Vec<(String, String, Weight)>, HashMap<String, Weight>),
      Box<dyn Error + 'static>
  > {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let focus_id = info.node_name_to_id_locked(focus)?;

    let focus_vector : Vec<(NodeId, NodeId, Weight)> =
      rank
        .neighbors_weighted(focus_id, true).unwrap().iter()
        .chain(rank.neighbors_weighted(focus_id, false).unwrap().iter())
        .map(|(target_id, weight)| (focus_id, *target_id, *weight))
        .collect();

    let mut copy = MyGraph::new();

    for (a_id, b_id, w_ab) in focus_vector {
      let b = info.node_id_to_name_locked(b_id)?;

      if b.starts_with("U") {
        if positive_only {
          let score = match rank.get_node_score(a_id, b_id) {
            Ok(x) => x,
            Err(MeritRankError::NodeIsNotCalculated) => {
              rank.calculate(a_id, *NUM_WALK)?;
              rank.get_node_score(a_id, b_id)?
            },
            Err(x) => {
              return Err(x.into());
            }
          };
          if score <= 0f64 {
            continue;
          }
        }

        let _ = copy.upsert_edge_with_nodes(a_id, b_id, w_ab)?;
      } else if b.starts_with("C") || b.starts_with("B") {
        // ? # For connections user-> comment | beacon -> user,
        // ? # convolve those into user->user

        let v_b : Vec<(NodeId, NodeId, Weight)> =
          rank
            .neighbors_weighted(b_id, true).unwrap().iter()
            .chain(rank.neighbors_weighted(b_id, false).unwrap().iter())
            .map(|(target_id, weight)| (b_id, *target_id, *weight))
            .collect();

        for (_, c_id, w_bc) in v_b {
          if positive_only && w_bc <= 0.0f64 {
            continue;
          }
          if c_id == a_id || c_id == b_id { // note: c_id==b_id not in Python version !?
            continue;
          }

          let c = info.node_id_to_name_locked(c_id)?;

          if !c.starts_with("U") {
            continue;
          }
          // let w_ac = self.get_transitive_edge_weight(a, b, c);
          // TODO: proper handling of negative edges
          // Note that enemy of my enemy is not my friend.
          // Though, this is pretty irrelevant for our current case
          // where comments can't have outgoing negative edges.
          // return w_ab * w_bc * (-1 if w_ab < 0 and w_bc < 0 else 1)
          let w_ac = w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });
          let _ = copy.upsert_edge_with_nodes(a_id, c_id, w_ac)?;
        }
      }
    }

    // self.remove_outgoing_edges_upto_limit(G, ego, focus, limit or 3):
    // neighbours = list(dest for src, dest in G.out_edges(focus))

    let neighbours : Vec<(EdgeIndex, NodeIndex, NodeId)> = copy.outgoing(focus_id);
    let ego_id = info.node_name_to_id_locked(ego)?;

    let mut sorted: Vec<(Weight, (&EdgeIndex, &NodeIndex))> =
      neighbours
        .iter()
        .map(|(edge_index, node_index, node_id)| {
          let w: f64 = rank.get_node_score(ego_id, *node_id).unwrap_or(0f64);
          (w, (edge_index, node_index))
        })
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    //sort by weight

    // for dest in sorted(neighbours, key=lambda x: self.get_node_score(ego, x))[limit:]:
    let limited: Vec<&(&EdgeIndex, &NodeIndex)> =
      sorted.iter()
        .map(|(_, tuple)| tuple)
        .take(limit.try_into().unwrap())
        .collect();

    for (_edge_index, node_index) in limited {
      let node_id = copy.index2node(**node_index);
      copy.remove_edge(ego_id, node_id);
      //G.remove_node(dest) // ???
    }

    // add_path_to_graph(G, ego, focus)
    let path: Vec<NodeId> =
      copy
        .shortest_path(ego_id, focus_id)
        .unwrap_or(Vec::new());
    // add_path_to_graph(G, ego, focus)
    // Note: no loops or "self edges" are expected in the path

    let v3: Vec<&NodeId> = path.iter().take(limit.try_into().unwrap()).collect::<Vec<&NodeId>>(); // was: (3)
    if let Some((&a, &b, &c)) = v3.clone().into_iter().collect_tuple() {
      // # merge transitive edges going through comments and beacons

      // ???
      /*
      if c is None and not (a.startswith("C") or a.startswith("B")):
        new_edge = (a, b, self.get_edge(a, b))
      elif ... */

      let a_name = info.node_id_to_name_locked(a)?;
      let b_name = info.node_id_to_name_locked(b)?;
      if b_name.starts_with("C") || b_name.starts_with("B") {
        let w_ab = copy.edge_weight(a, b).unwrap();
        let w_bc = copy.edge_weight(b, c).unwrap();

        // get_transitive_edge_weight
        let w_ac: f64 = w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });
        copy.upsert_edge(a, c, w_ac)?;
      } else if a_name.starts_with("U") {
        let weight = copy.edge_weight(a, b).unwrap();
        copy.upsert_edge(a, b, weight)?;
      }
    } else if let Some((&a, &b)) = v3.clone().into_iter().collect_tuple() {
      /*
      # Add the final (and only)
      final_nodes = ego_to_focus_path[-2:]
      final_edge = (*final_nodes, self.get_edge(*final_nodes))
      edges.append(final_edge)
      */
      // ???
      let weight = copy.edge_weight(a, b).unwrap();
      copy.upsert_edge(a, b, weight)?;
    } else if v3.len() == 1 {
      // ego == focus ?
      // do nothing
    } else if v3.is_empty() {
      // No path found, so add just the focus node to show at least something
      //let node = mrgraph::meritrank::node::Node::new(focus_id);
      copy.add_node(focus_id);
    } else {
      return Err("Gravity graph failure".into());
    }

    // self.remove_self_edges(copy);
    // todo: just not let them pass into the graph

    let (nodes, edges) = copy.all();

    let table: Vec<(String, String, f64)> =
      edges
        .iter()
        .map(|(n1, n2, weight)| {
          let name1 = info.node_id_to_name_locked(*n1).unwrap();
          let name2 = info.node_id_to_name_locked(*n2).unwrap();
          (name1, name2, *weight)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let nodes_dict: HashMap<String, Weight> =
      nodes
        .iter()
        .map(|node_id| {
          let name = info.node_id_to_name_locked(*node_id).unwrap();

          if !rank.get_personal_hits().contains_key(&ego_id) {
            let _ = rank.calculate(ego_id, *NUM_WALK).unwrap();
          }
          let score =
            rank.get_node_score(ego_id, *node_id).unwrap();
          (name, score)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<HashMap<String, Weight>>();

    Ok((table, nodes_dict))
  }

  fn mr_gravity_graph(
    &self,
    ego: &str,
    focus: &str,
    positive_only: bool,
    limit: Option<i32>
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let (result, _) = self.gravity_graph(ego, focus, positive_only, limit.unwrap_or(i32::MAX))?;
    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn mr_gravity_nodes(
    &self,
    ego: &str,
    focus: &str,
    positive_only: bool,
    limit: Option<i32>
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    // TODO: change HashMap to string pairs here!?
    let (_, hash_map) = self.gravity_graph(ego, focus, positive_only, limit.unwrap_or(i32::MAX))?;
    let result: Vec<_> = hash_map.iter().collect();
    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn get_connected(&self, ego : &str) -> Result<Vec<(String, String)>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let node_id = info.node_name_to_id_locked(ego)?;

    let result: Vec<(String, String)> =
      rank
        .neighbors_weighted(node_id, true).unwrap_or(HashMap::new()).iter()
        .chain(rank.neighbors_weighted(node_id, false).unwrap_or(HashMap::new()).iter())
        .map(|(target_id, _weight)| (
          info.node_id_to_name_locked(node_id).unwrap_or(node_id.to_string()),
          info.node_id_to_name_locked(*target_id).unwrap_or(target_id.to_string())
        ))
        .collect();

    return Ok(result);
  }

  fn mr_connected(
    &self,
    ego: &str
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let edges = self.get_connected(ego)?;
    if edges.is_empty() {
      return Err("No edges".into());
    }
    return Ok(rmp_serde::to_vec(&edges)?);
  }

  fn get_reduced_graph(&self) -> Result<Vec<(String, String, f64)>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let node_ids : HashMap<NodeId, String> =
      info.node_names.clone()
        .into_iter()
        .map(|(name, info)| (info.id, name))
        .collect();

    let users : Vec<(String, NodeInfo)> =
      info.node_names.clone()
        .into_iter()
        .filter(|(_, info)| info.kind == NodeType::NodeUser) // filter zero user?
        .map(|(name, info)| (name, info))
        .collect();

    if users.is_empty() {
      return Ok(Vec::new());
    }

    for (_, info) in users.iter() {
      rank.calculate(info.id, *NUM_WALK)?;
    }

    let edges : Vec<(NodeId, NodeId, Weight)> =
      users.into_iter()
        .map(|(_name, ego_info)| {
          let result: Vec<(NodeId, NodeId, Weight)> =
            rank.get_ranks(ego_info.id, None)?
            .into_iter()
            .map(|(node_id, score)| (ego_info.id, node_id, score))
            .filter(|(ego_id, node_id, score)|
              node_ids.get(node_id)
                .map(|node| (node.starts_with("U") || node.starts_with("B")) &&
                  *score > 0.0 &&
                  ego_id != node_id)
                .unwrap_or(false) // todo: log
            ).collect();
          Ok::<Vec<(NodeId, NodeId, Weight)>, MeritRankError>(result)
        })
        .filter_map(|res| res.ok())
        .flatten()
        .collect::<Vec<(NodeId, NodeId, Weight)>>();

    //let (_, edges) = my_graph.all(); // not optimal
    // Note:
    // Just eat errors in node_id_to_name_unsafe bellow.
    // Should we pass them out?
    let result : Vec<(String, String, f64)> =
      edges
        .iter()
        .filter(|(ego_id, dest_id, _)|
          ego_id != dest_id
          //  TODO
          //  filter if ego or dest is Zero here (?)
        )
        .map(|(ego_id, dest_id, weight)| {
          let ego = info.node_id_to_name_locked(*ego_id).unwrap();
          (ego, dest_id, weight)
        })
        .filter(|(ego, _dest_id, _weight)|
          ego.starts_with("U")
        )
        .map(|(ego, dest_id, weight)| {
          let dest = info.node_id_to_name_locked(*dest_id).unwrap();
          (ego, dest, *weight)
        })
        .filter(|(_ego, dest, _weight)|
          dest.starts_with("U") || dest.starts_with("B")
        )
        .collect();

    return Ok(result);
  }

  fn mr_nodes(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let result : Vec<String> =
      info.node_names
        .iter()
        .map(|(name, info)| (name.clone(), *info))
        .filter(|(_, info)|
          match rank.neighbors_weighted(info.id, true) {
            Some(x) => x.len() > 0,
            None    => false,
          } ||
          match rank.neighbors_weighted(info.id, false) {
            Some(x) => x.len() > 0,
            None    => false,
          }
        )
        .map(|(name, _)| name)
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn mr_edges(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).unwrap()
      },
      &mut graph.info);

    let mut v : Vec<(String, String, Weight)> = vec![];
    for (_, src) in info.node_names.iter() {
      let src_name = info.node_id_to_name_locked(src.id)?;
      for (dst, weight) in
        rank.neighbors_weighted(src.id, true).unwrap_or(HashMap::new()).iter()
          .chain(rank.neighbors_weighted(src.id, false).unwrap_or(HashMap::new()).iter()) {
        let dst_name = info.node_id_to_name_locked(*dst)?;
        v.push((src_name.clone(), dst_name, *weight));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  fn delete_from_zero(&self) -> Result<(), Box<dyn Error + 'static>> {
    let edges = self.get_connected(&ZERO_NODE)?;

    for (src, dst) in edges.iter() {
      let _ = self.mr_delete_edge(src, dst)?;
    }

    return Ok(());
  }

  fn top_nodes(&self) -> Result<Vec<(String, f64)>, Box<dyn Error + 'static>> {
    let reduced = self.get_reduced_graph()?;

    if reduced.is_empty() {
      return Err("Reduced graph empty".into());
    }

    let mut pr = Pagerank::<&String>::new();

    reduced
      .iter()
      .filter(|(source, target, _weight)|
        *source!=*ZERO_NODE && *target!=*ZERO_NODE
      )
      .for_each(|(source, target, _weight)| {
        // TODO: check weight
        pr.add_edge(source, target);
      });

    pr.calculate();

    let (nodes, scores): (Vec<&&String>, Vec<f64>) =
      pr
        .nodes()  // already sorted by score
        .into_iter()
        .take(*TOP_NODES_LIMIT)
        .into_iter()
        .unzip();

    let res = nodes
      .into_iter()
      .cloned()
      .cloned()
      .zip(scores)
      .collect::<Vec<_>>();

    if res.is_empty() {
      return Err("No top nodes".into());
    }

    return Ok(res);
  }

  fn mr_reset(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let mut graph = GRAPH.lock()?;

    if !self.context.is_empty() {
      return Err("Can only reset for all contexts".into());
    }

    graph.reset();

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
  }

  fn mr_zerorec(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    //  NOTE
    //  This func is not thread-safe.

    self.delete_from_zero()?;

    let nodes = self.top_nodes()?;

    for (name, amount) in nodes.iter() {
      let _ = self.mr_put_edge(ZERO_NODE.as_str(), name.as_str(), *amount)?;
    }

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
  }
}
