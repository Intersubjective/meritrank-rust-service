//  FIXME
//  Floating-point arithmetic for weight calculation will
//  break invariance.
//

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use std::thread;
use std::time::Duration;
use std::env::var;
use std::string::ToString;
use std::sync::MutexGuard;
use std::error::Error;
use petgraph::graph::{EdgeIndex, NodeIndex};
use nng::{Aio, AioResult, Context, Message, Protocol, Socket};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, IntMap, MeritRankError, Weight, NodeId, Neighbors};
use ctrlc;

//  ================================================
//
//    ...Previously called mrgraph
//
//  ================================================

lazy_static::lazy_static! {
  //  FIXME
  //  Use local object instead of a singleton.
  pub static ref GRAPH : Arc<Mutex<GraphSingleton>> =
    Arc::new(Mutex::new(GraphSingleton::new()
      .expect("Unable to create graph singleton")));
}

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub enum NodeKind {
  #[default]
  Unknown,
  User,
  Beacon,
  Comment,
}

pub struct NodesInfo {
  //  FIXME(Performance)
  //  Use parallel array of node names for O(0) lookup.
  //  Add node type attributes.

  node_names : HashMap<String, NodeId>,
  node_count : usize,
}

impl NodesInfo {
  pub fn new() -> NodesInfo {
    NodesInfo {
      node_names : HashMap::new(),
      node_count : 0,
    }
  }

  pub fn node_name_to_id_locked(&self, node_name: &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    match self.node_names.get(node_name) {
      Some(x) => Ok(*x),
      _       => Err("Node doesn't exist".into())
    }
  }

  pub fn node_id_to_name_locked(&self, node_id: NodeId) -> Result<String, Box<dyn Error + 'static>> {
    for (name, id) in self.node_names.iter() {
      if *id == node_id {
        return Ok(name.to_string());
      }
    }
    Err("Node not found".into())
  }
}

pub struct GraphSingleton {
  pub graph  : MeritRank<NodeKind>,                  // null-context
  pub graphs : HashMap<String, MeritRank<NodeKind>>, // contexted
  pub info   : NodesInfo,
}

fn kind_from_name(name : &str) -> NodeKind {
  match name.chars().nth(0) {
    Some('U') => NodeKind::User,
    Some('B') => NodeKind::Beacon,
    Some('C') => NodeKind::Comment,
    _         => NodeKind::Unknown,
  }
}

impl GraphSingleton {
  /// Constructor
  pub fn new() -> Result<GraphSingleton, Box<dyn Error + 'static>> {
    Ok(GraphSingleton {
      graph  : MeritRank::new(Graph::<NodeKind>::new())?,
      graphs : HashMap::new(),
      info   : NodesInfo::new(),
    })
  }

  pub fn reset(&mut self) -> Result<(), Box<dyn Error + 'static>> {
    self.graph  = MeritRank::new(Graph::new())?;
    self.graphs = HashMap::new();
    self.info   = NodesInfo::new();
    Ok(())
  }

  pub fn contexts() -> Result<Vec<String>, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.graphs.keys().map(|ctx| ctx.clone()).collect_vec())
  }

  pub fn add_node(node_name : &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    Ok(GRAPH.lock()?.add_node_id(node_name))
  }

  pub fn add_node_id(&mut self, node_name : &str) -> NodeId {
    if let Some(&n) = self.info.node_names.get(node_name) {
      n
    } else {
      let node_id = self.info.node_count;
      self.info.node_count += 1;
      self.info.node_names.insert(node_name.to_string(), node_id);
      self.graph.add_node(node_id, kind_from_name(&node_name));
      node_id
    }
  }

  pub fn add_node_id_contexted(&mut self, context : &str, node_name : &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    if let Some(&n) = self.info.node_names.get(node_name) {
      Ok(n)
    } else {
      let node_id = self.add_node_id(node_name); // create a node in null-context
      if !self.graphs.contains_key(context) {
        self.graphs.insert(context.to_string(), MeritRank::new(Graph::new())?);
      }
      let graph = self.graphs.get_mut(context).ok_or("No context")?;
      graph.add_node(node_id.into(), kind_from_name(&node_name));
      Ok(node_id)
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

  static ref EMPTY_RESULT : Vec<u8> = {
    const EMPTY_ROWS_VEC : Vec<(&str, &str, f64)> = Vec::new();
    rmp_serde::to_vec(&EMPTY_ROWS_VEC).expect("Unable to serialize empty result")
  };
}

const VERSION : &str = match option_env!("CARGO_PKG_VERSION") {
  Some(x) => x,
  None    => "dev",
};

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
  println!("Starting server {} at {}", VERSION, *SERVICE_URL);
  println!("NUM_WALK={}", *NUM_WALK);

  let s = Socket::new(Protocol::Rep0)?;
  s.listen(&SERVICE_URL)?;

  loop {
    let request: Message = s.recv()?;
    let reply: Vec<u8> = process(request);
    let _ = s.send(reply.as_slice()).map_err(|(_, e)| e)?;
  }
}

fn main_async(threads : usize) -> Result<(), Box<dyn Error + 'static>> {
  println!("Starting server {} at {}, {} threads", VERSION, *SERVICE_URL, threads);
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
    AioResult::Send(Ok(_)) => ctx.recv(&aio).expect("RECV failed"),

    // We successfully received a message.
    AioResult::Recv(Ok(req)) => {
      let msg: Vec<u8> = process(req);
      ctx.send(&aio, msg.as_slice()).expect("SEND failed");
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
      let s : String = e.to_string();
      rmp_serde::to_vec(&s).expect("Unable to serialize error string")
    })
}

fn mr_service() -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let s : String = VERSION.to_string();
  Ok(rmp_serde::to_vec(&s)?)
}

fn mr_node_score_null(ego : &str, target : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let mut graph = GRAPH.lock()?;

  let ego_id    = graph.info.node_name_to_id_locked(ego)?;
  let target_id = graph.info.node_name_to_id_locked(target)?;

  let mut w : Weight = 0.0;
  for (_, rank) in graph.graphs.iter_mut() {
    w += match rank.get_node_score(ego_id, target_id) {
      Ok(x) => x,
      _     => {
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        rank.get_node_score(ego_id, target_id)?
      }
    }
  }

  let result : Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
  Ok(rmp_serde::to_vec(&result)?)
}

fn mr_scores_null(ego : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  let  graph         =  &mut *GRAPH.lock()?;
  let (graphs, info) = (&mut graph.graphs, &mut graph.info);

  let ego_id = info.node_name_to_id_locked(ego)?;

  let result: Vec<_> =
    graphs
      .iter_mut()
      .filter_map(|(_context, rank)| {
        let rank_result = match rank.get_ranks(ego_id, None) {
          Ok(x) => x,
          _     => {
            let _ = rank.calculate(ego_id, *NUM_WALK).ok()?;
            rank.get_ranks(ego_id, None).ok()?
          }
        };
        let rows : Vec<_> =
          rank_result
            .into_iter()
            .map(|(n, s)| {
              (
                (
                  ego,
                  info.node_id_to_name_locked(n)
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
      self.mr_scores(ego, "",        false,         f64::MAX, true,  f64::MIN, true,  0,     u32::MAX)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, false,         score_lt, false, score_gt, false, 0,     u32::MAX)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">=", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, false,         score_lt, false, score_gt, true,  0,     u32::MAX)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, false,         score_lt, true,  score_gt, false, 0,     u32::MAX)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">=", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, false,         score_lt, true,  score_gt, true,  0,     u32::MAX)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, hide_personal, score_lt, false, score_gt, false, index, count)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">=", score_gt), ("score", "<", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, hide_personal, score_lt, false, score_gt, true,  index, count)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<=", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, hide_personal, score_lt, true,  score_gt, false, index, count)
    } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">=", score_gt), ("score", "<=", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(slice) {
      self.mr_scores(ego, node_kind, hide_personal, score_lt, true,  score_gt, true,  index, count)
    } else if let Ok((((subject, object, amount), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_put_edge(subject, object, amount)
    } else if let Ok(((("src", "delete", ego), ("dest", "delete", target)), ())) = rmp_serde::from_slice(slice) {
      self.mr_delete_edge(ego, target)
    } else if let Ok(((("src", "delete", ego), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_delete_node(ego)
    } else if let Ok((((ego, "gravity", focus), positive_only, index, count), ())) = rmp_serde::from_slice(slice) {
      self.mr_graph(ego, focus, positive_only, index, count)
    } else if let Ok((((ego, "connected"), ), ())) = rmp_serde::from_slice(slice) {
      self.mr_connected(ego)
    } else if let Ok(("nodes", ())) = rmp_serde::from_slice(slice) {
      self.mr_nodelist()
    } else if let Ok(("edges", ())) = rmp_serde::from_slice(slice) {
      self.mr_edges()
    } else if let Ok(("users_stats", ego, ())) = rmp_serde::from_slice(slice) {
      self.mr_mutual_scores(ego)
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
      Ok(x) => x,
      _     => {
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        rank.get_node_score(ego_id, target_id)?
      }
    };

    let result: Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  fn mr_scores(
    &self,
    ego           : &str,
    kind_str      : &str,
    hide_personal : bool,
    score_lt      : f64,
    score_lte     : bool,
    score_gt      : f64,
    score_gte     : bool,
    index         : u32,
    count         : u32
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>>
  {
    let kind = match kind_str {
      ""  => NodeKind::Unknown,
      "U" => NodeKind::User,
      "B" => NodeKind::Beacon,
      "C" => NodeKind::Comment,
       _  => {
         return Err(format!("Invalid node kind \"{}\"; only \"U\", \"B\", \"C\" are allowed", kind_str).into());
      },
    };

    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let node_id = info.node_name_to_id_locked(ego)?;

    //  NOTE
    //  We don't have to recalculate scores since we use
    //  incremental MeritRank.

    let ranks = match rank.get_ranks(node_id, None) {
      Ok(x) => x,
      _     => {
        let _ = rank.calculate(node_id, *NUM_WALK)?;
        rank.get_ranks(node_id, None)?
      }
    };

    let mut intermediate : Vec<(NodeId, NodeKind, Weight)> =
      ranks
        .into_iter()
        .map(|(n, w)| {
          (
            n,
            rank.get_node_data(n).unwrap_or(NodeKind::Unknown),
            w,
          )
        })
        .filter(|(_, target_kind, _)| {
          match kind {
            NodeKind::Unknown => true,
            _                 => kind == *target_kind,
          }
        })
        .filter(|(_, _, score)| score_gt < *score || (score_gte && score_gt == *score))
        .filter(|(_, _, score)| *score < score_lt || (score_lte && score_lt == *score))
        .filter(|(target_id, target_kind, _)|
          if hide_personal {
            !((*target_kind == NodeKind::Comment || *target_kind == NodeKind::Beacon) &&
              rank.get_edge(*target_id, node_id).is_some())
          } else {
            true
          }
        )
        .collect();

    intermediate.sort_by(|(_, _, a), (_, _, b)| a.total_cmp(b));

    let page : Result<Vec<(&str, String, Weight)>, _> =
      intermediate
        .iter()
        .skip(index as usize)
        .take(count as usize)
        .map(|(target_id, _, weight)| {
          match info.node_id_to_name_locked(*target_id) {
            Ok(x)  => Ok((ego, x, *weight)),
            Err(x) => Err(x)
          }
        })
        .collect();

    Ok(rmp_serde::to_vec(&page?)?)
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
        graph.graphs.insert(self.context.clone(), MeritRank::new(Graph::new())?);
      }
      let null_weight    = graph.graph.get_edge(src, dst).unwrap_or(0.0);
      let contexted_rank = graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?;
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
        ( graph.add_node_id_contexted(self.context.as_str(), src)?,
          graph.add_node_id_contexted(self.context.as_str(), dst)? )
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
    let     id    = *graph.info.node_names.get(ego).ok_or("No node")?;

    for n in graph.graph.neighbors_weighted(id, Neighbors::All).ok_or("Unable to get neighbors")?.keys() {
      self.set_edge_locked(&mut graph, id, *n, 0.0)?;
    }

    Ok(EMPTY_RESULT.to_vec())
  }

  fn gravity_graph(
    &self,
    ego           : &str,
    focus         : &str,
    positive_only : bool
  ) -> Result<
      Vec<(String, String, Weight)>,
      Box<dyn Error + 'static>
  > {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let focus_id = info.node_name_to_id_locked(focus)?;

    let focus_vector : Vec<(NodeId, NodeId, Weight)> =
      rank
        .neighbors_weighted(focus_id, Neighbors::All).ok_or("Unable to get neighbors")?.iter()
        .map(|(target_id, weight)| (focus_id, *target_id, *weight))
        .collect();

    let mut copy = Graph::new();

    for (a_id, b_id, w_ab) in focus_vector {
      let b_kind = rank.get_node_data(b_id).unwrap_or(NodeKind::Unknown);

      if b_kind == NodeKind::User {
        if positive_only {
          let score = match rank.get_node_score(a_id, b_id) {
            Ok(x) => x,
            _ => {
              rank.calculate(a_id, *NUM_WALK)?;
              rank.get_node_score(a_id, b_id)?
            }
          };

          if score <= 0f64 {
            continue;
          }
        }

        let _ = copy.upsert_edge_with_nodes(
          a_id, NodeKind::Unknown,
          b_id, NodeKind::Unknown,
          w_ab
        )?;
      } else if b_kind == NodeKind::Comment || b_kind == NodeKind::Beacon {
        let v_b : Vec<(NodeId, NodeId, Weight)> =
          rank
            .neighbors_weighted(b_id, Neighbors::All).ok_or("Unable to get neighbors")?.iter()
            .map(|(target_id, weight)| (b_id, *target_id, *weight))
            .collect();

        for (_, c_id, w_bc) in v_b {
          if positive_only && w_bc <= 0.0f64 {
            continue;
          }
          if c_id == a_id || c_id == b_id {
            continue;
          }

          let c_kind = rank.get_node_data(c_id).unwrap_or(NodeKind::Unknown);

          if c_kind != NodeKind::User {
            continue;
          }

          let w_ac = w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });
          let _ = copy.upsert_edge_with_nodes(
            a_id, NodeKind::Unknown,
            c_id, NodeKind::Unknown,
            w_ac
          )?;
        }
      }
    }

    let neighbors : Vec<(EdgeIndex, NodeIndex, NodeId)> = copy.outgoing(focus_id);
    let ego_id = info.node_name_to_id_locked(ego)?;

    let mut sorted = Vec::<(Weight, (EdgeIndex, NodeIndex))>::new();
    sorted.reserve_exact(neighbors.len());

    for (edge_index, node_index, node_id) in neighbors {
      let weight = match rank.get_node_score(ego_id, node_id) {
        Ok(x) => x,
        _     => {
          assert!(false);
          let _ = rank.calculate(ego_id, *NUM_WALK)?;
          rank.get_node_score(ego_id, node_id)?
        }
      };
      sorted.push((weight, (edge_index, node_index)));
    }

    sorted.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    let limited : Vec<(EdgeIndex, NodeIndex)> =
      sorted.iter()
        .map(|(_, tuple)| *tuple)
        .collect();

    for (_edge_index, node_index) in limited {
      let node_id = copy.index2node(node_index);
      copy.remove_edge(ego_id, node_id);
    }

    let path: Vec<NodeId> =
      copy
        .shortest_path(ego_id, focus_id)
        .unwrap_or(Vec::new());

    let v3 : Vec<&NodeId> =
      path
        .iter()
        .collect::<Vec<&NodeId>>(); // was: (3)

    if let Some((&a, &b, &c)) = v3.clone().into_iter().collect_tuple() {
      let a_kind = rank.get_node_data(a).unwrap_or(NodeKind::Unknown);
      let b_kind = rank.get_node_data(b).unwrap_or(NodeKind::Unknown);

      if b_kind == NodeKind::Comment || b_kind == NodeKind::Beacon {
        let w_ab = copy.edge_weight(a, b).ok_or("Unable to get edge weight")?;
        let w_bc = copy.edge_weight(b, c).ok_or("Unable to get edge weight")?;

        let w_ac: f64 = w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });
        copy.upsert_edge(a, c, w_ac)?;
      } else if a_kind == NodeKind::User {
        let weight = copy.edge_weight(a, b).ok_or("Unable to get edge weight")?;
        copy.upsert_edge(a, b, weight)?;
      }
    } else if let Some((&a, &b)) = v3.clone().into_iter().collect_tuple() {
      let weight = copy.edge_weight(a, b).ok_or("Unable to get edge weight")?;
      copy.upsert_edge(a, b, weight)?;
    } else if v3.len() == 1 {
    } else if v3.is_empty() {
      copy.add_node(focus_id, NodeKind::Unknown);
    } else {
      return Err("Gravity graph failure".into());
    }

    let (_, edges) = copy.all();

    edges
      .iter()
      .map(|(n1, n2, weight)| {
        match (
          info.node_id_to_name_locked(*n1),
          info.node_id_to_name_locked(*n2)
        ) {
          (Ok(name1), Ok(name2)) => Ok((name1, name2, *weight)),
          _                      => Err("No node".into())
        }
      })
      .collect()
  }

  fn mr_graph(
    &self,
    ego           : &str,
    focus         : &str,
    positive_only : bool,
    index         : u32,
    count         : u32
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let edges            = self.gravity_graph(ego, focus, positive_only)?;
    let result : Vec<_>  = edges.iter().skip(index as usize).take(count as usize).collect();
    let v      : Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn get_connected(&self, ego : &str) -> Result<Vec<(String, String)>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let node_id = info.node_name_to_id_locked(ego)?;

    let result: Vec<(String, String)> =
      rank
        .neighbors_weighted(node_id, Neighbors::All).unwrap_or(IntMap::default()).iter()
        .map(|(target_id, _weight)| (
          info.node_id_to_name_locked(node_id).unwrap_or(node_id.to_string()),
          info.node_id_to_name_locked(*target_id).unwrap_or(target_id.to_string())
        ))
        .collect();

    return Ok(result);
  }

  fn mr_connected(
    &self,
    ego   : &str
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let edges = self.get_connected(ego)?;
    if edges.is_empty() {
      return Err("No edges".into());
    }
    return Ok(rmp_serde::to_vec(&edges)?);
  }

  fn recalculate_all(&self, num_walk : usize) -> Result<(), Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    for (_, id) in info.node_names.iter() {
      if rank.get_node_data(*id)? == NodeKind::User {
        rank.calculate(*id, num_walk)?;
      }
    }

    Ok(())
  }

  fn get_reduced_graph(&self) -> Result<Vec<(NodeId, NodeId, f64)>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let zero = info.node_name_to_id_locked(ZERO_NODE.as_str());

    let node_kinds : HashMap<NodeId, NodeKind> =
      info.node_names
        .iter()
        .map(|(_, id)| (*id, rank.get_node_data(*id).unwrap_or(NodeKind::Unknown)))
        .collect();

    let users : Vec<NodeId> =
      info.node_names
        .iter()
        .filter(|(_, id)| {
          match zero {
            Ok(x) => {
              if **id == x {
                return false;
              }
            }
            _ => {}
          }
          return match rank.get_node_data(**id) {
            Ok(NodeKind::User) => true,
            _                  => false,
          };
        })
        .map(|(_, id)| *id)
        .collect();

    if users.is_empty() {
      return Ok(Vec::new());
    }

    for id in users.iter() {
      rank.calculate(*id, *NUM_WALK)?;
    }

    let edges : Vec<(NodeId, NodeId, Weight)> =
      users.into_iter()
        .map(|id| {
          let result : Vec<(NodeId, NodeId, Weight)> =
            rank.get_ranks(id, None)?
            .into_iter()
            .map(|(node_id, score)| (id, node_id, score))
            .filter(|(ego_id, node_id, score)|
              node_kinds.get(node_id)
                .map(|kind| (*kind == NodeKind::User || *kind == NodeKind::Beacon) &&
                  *score > 0.0 &&
                  ego_id != node_id)
                .unwrap_or(false)
            ).collect();
          Ok::<Vec<(NodeId, NodeId, Weight)>, MeritRankError>(result)
        })
        .filter_map(|res| res.ok())
        .flatten()
        .collect();

    //let (_, edges) = my_graph.all(); // not optimal
    // Note:
    // Just eat errors in node_id_to_name_unsafe bellow.
    // Should we pass them out?
    let result : Vec<(NodeId, NodeId, f64)> =
      edges
        .iter()
        .filter(|(ego_id, dst_id, _)| {
          match zero {
            Ok(x) => {
              if *ego_id == x || *dst_id == x {
                return false;
              }
            },
            _ => {}
          }
          let ego_kind = *node_kinds.get(ego_id).unwrap_or(&NodeKind::Unknown);
          let dst_kind = *node_kinds.get(dst_id).unwrap_or(&NodeKind::Unknown);
          return  ego_id != dst_id &&
                  ego_kind == NodeKind::User &&
                 (dst_kind == NodeKind::User || dst_kind == NodeKind::Beacon);
        })
        .map(|(ego_id, dst_id, weight)| (*ego_id, *dst_id, *weight))
        .collect();

    return Ok(result);
  }

  fn mr_nodelist(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (_rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let result : Vec<(String,)> =
      info.node_names
        .iter()
        //.map(|(name, id)| (name.clone(), *id))
        //.filter(|(_, id)|
        //  match rank.neighbors_weighted(*id, Neighbors::All) {
        //    Some(x) => x.len() > 0,
        //    None    => false,
        //  }
        //)
        .map(|(name, _)| (name.clone(),))
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn mr_edges(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let mut v : Vec<(String, String, Weight)> = vec![];
    for (_, src) in info.node_names.iter() {
      let src_name = info.node_id_to_name_locked(*src)?;
      for (dst, weight) in
        rank.neighbors_weighted(*src, Neighbors::All).unwrap_or(IntMap::default()).iter() {
        let dst_name = info.node_id_to_name_locked(*dst)?;
        v.push((src_name.clone(), dst_name, *weight));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  fn mr_mutual_scores(&self, ego : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    let graph = &mut *GRAPH.lock()?;

    let (rank, info) = (if self.context.is_empty() {
        &mut graph.graph
      } else {
        graph.graphs.get_mut(self.context.as_str()).ok_or("No context")?
      },
      &mut graph.info);

    let ego_id = info.node_name_to_id_locked(ego)?;

    let ranks = match rank.get_ranks(ego_id, None) {
      Ok(x) => x,
      _     => {
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        rank.get_ranks(ego_id, None)?
      }
    };

    let mut v = Vec::<(String, Weight, Weight)>::new();

    v.reserve_exact(ranks.len());

    for (node, score) in ranks {
      if score > 0.0 && rank.get_node_data(node).unwrap_or(NodeKind::Unknown) == NodeKind::User {
        v.push((
          info.node_id_to_name_locked(node)?,
          score,
          match rank.get_node_score(node, ego_id) {
            Ok(x) => x,
            _     => {
              let _ = rank.calculate(node, *NUM_WALK)?;
              rank.get_node_score(node, ego_id)?
            }
          }
        ));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  fn delete_from_zero(&self) -> Result<(), Box<dyn Error + 'static>> {
    let edges = match self.get_connected(&ZERO_NODE) {
      Ok(x) => x,
      _     => return Ok(()),
    };

    for (src, dst) in edges.iter() {
      let _ = self.mr_delete_edge(src, dst)?;
    }

    return Ok(());
  }

  fn top_nodes(&self) -> Result<Vec<(NodeId, f64)>, Box<dyn Error + 'static>> {
    let reduced = self.get_reduced_graph()?;

    if reduced.is_empty() {
      return Err("Reduced graph empty".into());
    }

    let mut pr = Pagerank::<NodeId>::new();

    let zero = GraphSingleton::node_name_to_id(ZERO_NODE.as_str());

    reduced
      .iter()
      .filter(|(source, target, _weight)|
        match zero {
          Ok(x) => return *source != x && *target != x,
          _     => return true,
        }
      )
      .for_each(|(source, target, _weight)| {
        // TODO: check weight
        pr.add_edge(*source, *target);
      });

    pr.calculate();

    let (nodes, scores): (Vec<NodeId>, Vec<f64>) =
      pr
        .nodes()  // already sorted by score
        .into_iter()
        .take(*TOP_NODES_LIMIT)
        .into_iter()
        .unzip();

    let res = nodes
      .into_iter()
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

    let _ = graph.reset()?;

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
  }

  fn mr_zerorec(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    //  NOTE
    //  This func is not thread-safe.

    self.recalculate_all(0)?; // FIXME Ad hok hack
    self.delete_from_zero()?;

    let nodes = self.top_nodes()?;

    self.recalculate_all(0)?; // FIXME Ad hok hack
    {
      let mut graph = GRAPH.lock()?;

      let zero = match graph.info.node_name_to_id_locked(ZERO_NODE.as_str()) {
        Ok(x) => x,
        _     => graph.add_node_id(ZERO_NODE.as_str()),
      };

      for (node_id, amount) in nodes.iter() {
        self.set_edge_locked(&mut graph, zero, *node_id, *amount)?;
      }
    }
    self.recalculate_all(*NUM_WALK)?; // FIXME Ad hok hack

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
  }
}
