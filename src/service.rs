use std::{
  sync::atomic::{AtomicBool, Ordering},
  collections::HashMap,
  sync::{Arc, Mutex, Condvar},
  ops::DerefMut,
  env::var,
  string::ToString,
  error::Error,
  thread
};
use chrono;
use itertools::Itertools;
use nng::{Aio, AioResult, Context, Protocol, Socket};
use petgraph::{visit::EdgeRef, graph::{DiGraph, NodeIndex}};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, IntMap, NodeId, Neighbors, MeritRankError};

use crate::commands::*;
use crate::astar::astar::*;

pub use meritrank::Weight;

//  ================================================================
//
//    Global options
//
//  ================================================================

pub static ERROR   : AtomicBool = AtomicBool::new(true);
pub static WARNING : AtomicBool = AtomicBool::new(true);
pub static INFO    : AtomicBool = AtomicBool::new(true);
pub static VERBOSE : AtomicBool = AtomicBool::new(true);
pub static TRACE   : AtomicBool = AtomicBool::new(true);

const VERSION : &str = match option_env!("CARGO_PKG_VERSION") {
  Some(x) => x,
  None    => "dev",
};

lazy_static::lazy_static! {
  pub static ref THREADS : usize =
    var("MERITRANK_SERVICE_THREADS")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(1);

  pub static ref ZERO_NODE : String =
    var("MERITRANK_ZERO_NODE")
      .unwrap_or("U000000000000".to_string());

  static ref SERVICE_URL : String =
    var("MERITRANK_SERVICE_URL")
      .unwrap_or("tcp://127.0.0.1:10234".to_string());

  static ref NUM_WALK : usize =
    var("MERITRANK_NUM_WALK")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(10000);

  static ref TOP_NODES_LIMIT : usize =
    var("MERITRANK_TOP_NODES_LIMIT")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(100);
}

//  ================================================================
//
//    Basic declarations
//
//  ================================================================

type BoxedError = Box<dyn Error + 'static>;

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub enum NodeKind {
  #[default]
  Unknown,
  User,
  Beacon,
  Comment,
}

#[derive(PartialEq, Eq, Clone, Default)]
pub struct NodeInfo {
  pub kind : NodeKind,
  pub name : String,
}

//  Augmented multi-context graph
//
#[derive(Clone)]
pub struct AugMultiGraph {
  pub node_count : usize,
  pub node_infos : Vec<NodeInfo>,
  pub node_ids   : HashMap<String, NodeId>,
  pub contexts   : HashMap<String, MeritRank<()>>,
}

#[derive(Clone)]
pub struct Command {
  pub id      : String,
  pub context : String,
  pub payload : Vec<u8>,
}

#[derive(Clone)]
pub struct Data {
  pub graph_readable : Arc<Mutex<AugMultiGraph>>,
  pub graph_writable : Arc<Mutex<AugMultiGraph>>,
  pub queue_commands : Arc<Mutex<Vec<Command>>>,
  pub cond_add       : Arc<Condvar>,
  pub cond_done      : Arc<Condvar>,
}

//  ================================================================
//
//    Utils
//
//  ================================================================

static LOG_MUTEX : Mutex<()> = Mutex::new(());

fn log_with_time(prefix : &str, message : &str) {
  let time      = chrono::offset::Local::now();
  let time_str  = time.format("%Y-%m-%d %H:%M:%S");
  let millis    = time.timestamp_millis() % 1000;
  let thread_id = thread::current().id();

  match LOG_MUTEX.lock() {
    Ok(_) => {
      println!("{}.{:03} {:3?}  {}{}", time_str, millis, thread_id, prefix, message);
    },
    _ => {
      println!("{}.{:03} {:3?}  LOG MUTEX FAILED", time_str, millis, thread_id);
    },
  };
}


macro_rules! log_error {
  ($($arg:expr),*) => {
    if ERROR.load(Ordering::Relaxed) {
      log_with_time("ERROR   ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! error {
  ($func:expr, $($arg:expr),*) => {
    {
      if ERROR.load(Ordering::Relaxed) {
        log_with_time(format!("ERROR   ({}) ", $func).as_str(), format!($($arg),*).as_str());
      }
      Err(format!($($arg),*).into())
    }
  };
}

#[allow(unused_macros)]
macro_rules! log_warning {
  ($($arg:expr),*) => {
    if WARNING.load(Ordering::Relaxed) {
      log_with_time("WARNING ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_info {
  ($($arg:expr),*) => {
    if INFO.load(Ordering::Relaxed) {
      log_with_time("INFO    ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_verbose {
  ($($arg:expr),*) => {
    if VERBOSE.load(Ordering::Relaxed) {
      log_with_time("VERBOSE --- ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_trace {
  ($($arg:expr),*) => {
    if TRACE.load(Ordering::Relaxed) {
      log_with_time("TRACE   --- --- ", format!($($arg),*).as_str());
    }
  };
}

fn kind_from_name(name : &str) -> NodeKind {
  log_trace!("kind_from_name: `{}`", name);

  match name.chars().nth(0) {
    Some('U') => NodeKind::User,
    Some('B') => NodeKind::Beacon,
    Some('C') => NodeKind::Comment,
    _         => NodeKind::Unknown,
  }
}

fn get_score_or_recalculate(
  graph   : &mut MeritRank<()>,
  src_id  : NodeId,
  dst_id  : NodeId
) -> Result<Weight, BoxedError> {
  log_trace!("get_score_or_recalculate");

  match graph.get_node_score(src_id, dst_id) {
    Ok(score) => Ok(score),
    Err(MeritRankError::NodeDoesNotExist) => {
      log_warning!("Node does not exist: {}, {}", src_id, dst_id);
      Ok(0.0)
    },
    _ => {
      log_warning!("Recalculating node {}", src_id);
      match graph.calculate(src_id, *NUM_WALK) {
        Err(e) => {
          log_warning!("{}", e);
          return Ok(0.0);
        },
        _ => {},
      };
      Ok(graph.get_node_score(src_id, dst_id)?)
    },
  }
}

fn get_ranks_or_recalculate(
  graph   : &mut MeritRank<()>,
  node_id : NodeId
) -> Result<Vec<(NodeId, Weight)>, BoxedError> {
  log_trace!("get_ranks_or_recalculate");

  match graph.get_ranks(node_id, None) {
    Ok(ranks) => Ok(ranks),
    Err(MeritRankError::NodeDoesNotExist) => {
      log_warning!("Node does not exist: {}", node_id);
      Ok(vec![])
    },
    _ => {
      log_warning!("Recalculating node: {}", node_id);
      match graph.calculate(node_id, *NUM_WALK) {
        Err(e) => {
          log_warning!("{}", e);
          return Ok(vec![]);
        },
        _ => {},
      };
      Ok(graph.get_ranks(node_id, None)?)
    },
  }
}

impl Default for AugMultiGraph {
  fn default() -> AugMultiGraph {
    AugMultiGraph::new().expect("Unable to create AugMultiGraph")
  }
}

impl AugMultiGraph {
  pub fn new() -> Result<AugMultiGraph, BoxedError> {
    log_trace!("AugMultiGraph::new");

    Ok(AugMultiGraph {
      node_count   : 0,
      node_infos   : Vec::new(),
      node_ids     : HashMap::new(),
      contexts     : HashMap::new(),
    })
  }

  fn copy_from(&mut self, other : &AugMultiGraph) {
    self.node_count = other.node_count;
    self.node_infos = other.node_infos.clone();
    self.node_ids   = other.node_ids.clone();
    self.contexts   = other.contexts.clone();
  }

  fn reset(&mut self) -> Result<(), BoxedError> {
    log_trace!("reset");

    self.node_count   = 0;
    self.node_infos   = Vec::new();
    self.node_ids     = HashMap::new();
    self.contexts     = HashMap::new();

    Ok(())
  }

  fn node_id_from_name(&self, node_name : &str) -> Result<NodeId, BoxedError> {
    log_trace!("node_id_from_name");

    match self.node_ids.get(node_name) {
      Some(x) => Ok(*x),
      _       => {
        error!("node_id_from_name", "Node does not exist: `{}`", node_name)
      },
    }
  }

  fn node_info_from_id(&self, node_id : NodeId) -> Result<&NodeInfo, BoxedError> {
    log_trace!("node_info_from_id: {}", node_id);

    match self.node_infos.get(node_id) {
      Some(x) => Ok(x),
      _       => {
        error!("node_info_from_id", "Node does not exist: `{}`", node_id)
      },
    }
  }

  //  Get mutable graph from a context
  //
  fn graph_from(&mut self, context : &str) -> Result<&mut MeritRank<()>, BoxedError> {
    log_trace!("graph_from: `{}`", context);

    if !self.contexts.contains_key(context) {
      if context.is_empty() {
        log_verbose!("Add context: NULL");
      } else {
        log_verbose!("Add context: `{}`", context);
      }
      self.contexts.insert(context.to_string(), MeritRank::new(Graph::new())?);
    }

    match self.contexts.get_mut(context) {
      Some(graph) => Ok(graph),
      None        => {
        error!("graph_from", "Unable to add context `{}`", context)
      },
    }
  }

  fn find_or_add_node_by_name(
    &mut self,
    context   : &str,
    node_name : &str
  ) -> Result<NodeId, BoxedError> {
    log_trace!("find_or_add_node_by_name: `{}`, `{}`", context, node_name);

    if let Some(&node_id) = self.node_ids.get(node_name) {
      Ok(node_id)
    } else {
      let node_id = self.node_count;
      self.node_count += 1;
      self.node_infos.resize(self.node_count, NodeInfo::default());
      self.node_infos[node_id] = NodeInfo {
        kind : kind_from_name(&node_name),
        name : node_name.to_string(),
      };
      self.node_ids.insert(node_name.to_string(), node_id);

      if !context.is_empty() {
        log_verbose!("Add node in NULL: {}", node_id);
        self.graph_from("")?.add_node(node_id, ());
      }

      log_verbose!("Add node in `{}`: {}", context, node_id);
      self.graph_from(context)?.add_node(node_id, ());

      Ok(node_id)
    }
  }

  fn set_edge(
    &mut self,
    context : &str,
    src     : NodeId,
    dst     : NodeId,
    amount  : f64
  ) -> Result<(), BoxedError> {
    log_trace!("set_edge: `{}` `{}` `{}` {}", context, src, dst, amount);

    if context.is_empty() {
      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, amount);
      self.graph_from("")?.add_edge(src, dst, amount);
    } else {
      //  This doesn't make any sense but it's Rust.

      let null_weight = self.graph_from("")?     .get_edge(src, dst).unwrap_or(0.0);
      let old_weight  = self.graph_from(context)?.get_edge(src, dst).unwrap_or(0.0);
      let delta       = null_weight + amount - old_weight;

      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, delta);
      self.graph_from("")?.add_edge(src, dst, delta);

      log_verbose!("Add edge in `{}`: {} -> {} for {}", context, src, dst, amount);
      self.graph_from(context)?.add_edge(src, dst, amount);
    }

    Ok(())
  }

  fn connected_nodes(
    &mut self,
    context   : &str,
    ego       : NodeId
  ) -> Result<
    Vec<(NodeId, NodeId)>,
    BoxedError
  > {
    log_trace!("connected_nodes: `{}` {}", context, ego);

    let edge_ids : Vec<(NodeId, NodeId)> =
      self.graph_from(context)?
        .neighbors_weighted(ego, Neighbors::All).ok_or("Node does not exist")?.iter()
        .map(|(dst_id, _weight)| (
          ego,
          *dst_id
        ))
        .collect();

    Ok(edge_ids)
  }

  fn connected_node_names(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<
    Vec<(&str, &str)>,
    BoxedError
  > {
    log_trace!("connected_node_names: `{}` `{}`", context, ego);

    let src_id = self.node_id_from_name(ego)?;

    let edge_ids = self.connected_nodes(context, src_id)?;

    let res : Result<Vec<(&str, &str)>, BoxedError> =
      edge_ids
        .into_iter()
        .map(|(src_id, dst_id)| Ok((
          self.node_info_from_id(src_id)?.name.as_str(),
          self.node_info_from_id(dst_id)?.name.as_str()
        )))
        .collect();

    Ok(res?)
  }

  fn recalculate_all(&mut self, num_walk : usize) -> Result<(), BoxedError> {
    log_trace!("recalculate_all: {}", num_walk);

    let infos = self.node_infos.clone();
    let graph = self.graph_from("")?;

    for id in 0..infos.len() {
      if infos[id].kind == NodeKind::User {
        graph.calculate(id, num_walk)?;
      }
    }

    Ok(())
  }
}

//  ================================================
//
//    Commands
//
//  ================================================

pub fn read_version() -> Result<Vec<u8>, BoxedError> {
  log_info!("CMD read_version");

  let s : String = VERSION.to_string();
  Ok(rmp_serde::to_vec(&s)?)
}

pub fn write_log_level(log_level : u32) -> Result<Vec<u8>, BoxedError> {
  log_info!("CMD write_log_level: {}", log_level);

  ERROR  .store(log_level > 0, Ordering::Relaxed);
  WARNING.store(log_level > 1, Ordering::Relaxed);
  INFO   .store(log_level > 2, Ordering::Relaxed);
  VERBOSE.store(log_level > 3, Ordering::Relaxed);
  TRACE  .store(log_level > 4, Ordering::Relaxed);

  Ok(rmp_serde::to_vec(&())?)
}

impl AugMultiGraph {
  pub fn read_node_score_null(&mut self, ego : &str, target : &str) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_node_score_null: `{}` `{}`", ego, target);

    let ego_id    = self.node_id_from_name(ego)?;
    let target_id = self.node_id_from_name(target)?;

    let mut w : Weight = 0.0;
    for (name, rank) in self.contexts.iter_mut() {
      if name.is_empty() {
        continue;
      }
      w += get_score_or_recalculate(rank, ego_id, target_id)?;
    }

    let result : Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  pub fn read_scores_null(&mut self, ego : &str) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_scores_null: `{}`", ego);

    let ego_id = self.node_id_from_name(ego)?;

    let intermediate : Vec<(&str, NodeId, Weight)> =
      self.contexts
        .iter_mut()
        .filter_map(|(context, rank)| {
          if context.is_empty() {
            return None;
          }
          let rank_result = get_ranks_or_recalculate(rank, ego_id).ok()?;
          let rows : Vec<_> =
            rank_result
              .into_iter()
              .map(|(node, score)| {
                ((ego, node), score)
              })
              .collect();
          Some(rows)
        })
        .flatten()
        .into_iter()
        .group_by(|(nodes, _)| nodes.clone())
        .into_iter()
        .map(|((src, dst), rows)|
          (src, dst, rows.map(|(_, score)| score).sum::<Weight>())
        )
        .collect();

    let result : Result<Vec<(&str, &str, Weight)>, BoxedError> =
      intermediate
        .iter()
        .map(|(ego, node, weight)|
          match self.node_info_from_id(*node) {
            Ok(info) => Ok((*ego, info.name.as_str(), *weight)),
            Err(x)   => error!("read_scores_null", "{}", x),
          }
        )
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result?)?;
    Ok(v)
  }

  pub fn read_node_score(
    &mut self,
    context : &str,
    ego     : &str,
    target  : &str
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_node_score: `{}` `{}` `{}`", context, ego, target);

    let ego_id    = self.node_id_from_name(ego)?;
    let target_id = self.node_id_from_name(target)?;

    let graph = self.graph_from(context)?;

    let w = get_score_or_recalculate(graph, ego_id, target_id)?;

    let result: Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  pub fn read_scores(
    &mut self,
    context       : &str,
    ego           : &str,
    kind_str      : &str,
    hide_personal : bool,
    score_lt      : f64,
    score_lte     : bool,
    score_gt      : f64,
    score_gte     : bool,
    index         : u32,
    count         : u32
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_scores: `{}` `{}` `{}` {} {} {} {} {} {} {}",
              context, ego, kind_str, hide_personal,
              score_lt, score_lte, score_gt, score_gte,
              index, count);

    let kind = match kind_str {
      ""  => NodeKind::Unknown,
      "U" => NodeKind::User,
      "B" => NodeKind::Beacon,
      "C" => NodeKind::Comment,
       _  => {
         return error!("read_scores", "Invalid node kind string: `{}`", kind_str);
      },
    };

    let node_id = self.node_id_from_name(ego)?;

    let ranks = get_ranks_or_recalculate(self.graph_from(context)?, node_id)?;

    let mut im : Vec<(NodeId, Weight)> =
      ranks
        .into_iter()
        .map(|(n, w)| (
          n,
          match self.node_info_from_id(n) {
            Ok(info) => info.kind,
            _        => NodeKind::Unknown
          },
          w,
        ))
        .filter(|(_, target_kind, _)| kind == NodeKind::Unknown || kind == *target_kind)
        .filter(|(_, _, score)| score_gt < *score || (score_gte && score_gt == *score))
        .filter(|(_, _, score)| *score < score_lt || (score_lte && score_lt == *score))
        .collect::<Vec<(NodeId, NodeKind, Weight)>>()
        .into_iter()
        .filter(|(target_id, target_kind, _)| {
          if !hide_personal || (*target_kind != NodeKind::Comment && *target_kind != NodeKind::Beacon) {
            return true;
          }
          match self.graph_from(context) {
            Ok(graph) => !graph.get_edge(*target_id, node_id).is_some(),
            Err(x)    => { log_error!("(read_scores) {}", x); false },
          }
        })
        .map(|(target_id, _, weight)| (target_id, weight))
        .collect();

    im.sort_by(|(_, a), (_, b)| a.total_cmp(b));

    let index = index as usize;
    let count = count as usize;

    let mut page : Vec<(&str, &str, Weight)> = vec![];
    page.reserve_exact(if count < im.len() { count } else { im.len() });

    for i in index..count {
      if i >= im.len() {
        break;
      }
      match self.node_info_from_id(im[i].0) {
        Ok(info) => { page.push((ego, info.name.as_str(), im[i].1)); },
        Err(e)   => { log_error!("(read_scores) {}", e); },
      }
    }

    Ok(rmp_serde::to_vec(&page)?)
  }

  pub fn write_put_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
    amount  : f64
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_put_edge: `{}` `{}` `{}` {}", context, src, dst, amount);

    let src_id = self.find_or_add_node_by_name(context, src)?;
    let dst_id = self.find_or_add_node_by_name(context, dst)?;

    self.set_edge(context, src_id, dst_id, amount)?;

    Ok(rmp_serde::to_vec(&())?)
  }

  pub fn write_delete_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_delete_edge: `{}` `{}` `{}`", context, src, dst);

    let src_id = self.node_id_from_name(src)?;
    let dst_id = self.node_id_from_name(dst)?;

    self.set_edge(context, src_id, dst_id, 0.0)?;

    Ok(rmp_serde::to_vec(&())?)
  }

  pub fn write_delete_node(
    &mut self,
    context : &str,
    node    : &str,
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_delete_node: `{}` `{}`", context, node);

    let id = self.node_id_from_name(node)?;

    for n in self.graph_from(context)?.neighbors_weighted(id, Neighbors::All).ok_or("Unable to get neighbors")?.keys() {
      self.set_edge(context, id, *n, 0.0)?;
    }

    Ok(rmp_serde::to_vec(&())?)
  }

  pub fn read_graph(
    &mut self,
    context       : &str,
    ego           : &str,
    focus         : &str,
    positive_only : bool,
    index         : u32,
    count         : u32
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_graph: `{}` `{}` `{}` {} {} {}",
              context, ego, focus, positive_only, index, count);

    let ego_id   = self.node_id_from_name(ego)?;
    let focus_id = self.node_id_from_name(focus)?;

    let mut indices  = HashMap::<NodeId, NodeIndex>::new();
    let mut ids      = HashMap::<NodeIndex, NodeId>::new();
    let mut im_graph = DiGraph::<NodeId, Weight>::new();

    {
      let index = im_graph.add_node(focus_id);
      indices.insert(focus_id, index);
      ids.insert(index, focus_id);
    }

    log_trace!("enumerate focus neighbors");

    match self.graph_from(context)?.neighbors_weighted(focus_id, Neighbors::All) {
      Some(focus_neighbors) => {
        for (dst_id, focus_dst_weight) in focus_neighbors {
          let dst_kind = self.node_info_from_id(dst_id)?.kind;

          if dst_kind == NodeKind::User {
            if positive_only && self.graph_from(context)?.get_edge(ego_id, dst_id).unwrap_or(0.0) <= 0.0 {
              continue;
            }

            if !indices.contains_key(&dst_id) {
              let index = im_graph.add_node(focus_id);
              indices.insert(dst_id, index);
              ids.insert(index, dst_id);
            }

            im_graph.add_edge(
              *indices.get(&focus_id).ok_or("Got invalid node")?,
              *indices.get(&dst_id).ok_or("Got invalid node")?,
              focus_dst_weight
            );
          } else if dst_kind == NodeKind::Comment || dst_kind == NodeKind::Beacon {
            let dst_neighbors = match self.graph_from(context)?.neighbors_weighted(dst_id, Neighbors::All) {
              Some(x) => x,
              _       => {
                continue;
              }
            };

            for (ngh_id, dst_ngh_weight) in dst_neighbors {
              if (positive_only && dst_ngh_weight <= 0.0) || ngh_id == focus_id || self.node_info_from_id(ngh_id)?.kind != NodeKind::User {
                continue;
              }

              let focus_ngh_weight = focus_dst_weight * dst_ngh_weight * if focus_dst_weight < 0.0 && dst_ngh_weight < 0.0 { -1.0 } else { 1.0 };

              if !indices.contains_key(&ngh_id) {
                let index = im_graph.add_node(ngh_id);
                indices.insert(ngh_id, index);
                ids.insert(index, ngh_id);
              }

              im_graph.add_edge(
                *indices.get(&focus_id).ok_or("Got invalid node")?,
                *indices.get(&ngh_id).ok_or("Got invalid node")?,
                focus_ngh_weight
              );
            }
          }
        }
      },
      None => {},
    }

    if ego_id == focus_id {
      log_trace!("ego is same as focus");
    } else {
      log_trace!("search shortest path");

      let graph_cloned = self.graph_from(context)?.graph.clone();

      //  ================================
      //
      //    A* search
      //

      let neighbor = |node : NodeId, index : usize| -> Result<Option<Link<NodeId, Weight>>, BoxedError> {
        let v : Vec<_> = graph_cloned.neighbors(node).into_iter().skip(index).take(1).collect();
        if v.is_empty() {
          Ok(None)
        } else {
          let n = v[0];
          let w = graph_cloned.edge_weight(node, n).ok_or("Got invalid edge")?;
          Ok(Some(
            Link::<NodeId, Weight> {
              neighbor       : n,
              exact_distance : if w.abs() < 0.001 { 1_000_000.0 } else { 1.0 / w },
            }
          ))
        }
      };

      let heuristic = |_node : NodeId| -> Result<Weight, BoxedError> {
        Ok(0.0)
        //  ad hok
        //Ok(((node as i64) - (focus_id as i64)).abs() as f64)
      };

      let mut astar_state = init(ego_id, focus_id, Weight::MAX);

      let mut status = Status::PROGRESS;
      let mut count  = 0;

      //  Do 10000 iterations max

      for _ in 0..10000 {
        count += 1;
        status = iteration(&mut astar_state, neighbor, heuristic)?;
        if status != Status::PROGRESS {
          break;
        }
      }

      log_trace!("did {} A* iterations", count);

      if status == Status::SUCCESS {
        log_trace!("path found");
      } else if status == Status::PROGRESS {
        log_error!("Unable to find a path from {} to {}", ego_id, focus_id);
      } else if status == Status::FAIL {
        log_error!("Path does not exist from {} to {}", ego_id, focus_id);
      }

      let ego_to_focus = path(&mut astar_state).ok_or("Unable to build a path")?;

      for node in ego_to_focus.iter() {
        log_trace!("path: {}", self.node_info_from_id(*node)?.name);
      }

      //  ================================

      let mut edges = Vec::<(NodeId, NodeId, Weight)>::new();
      edges.reserve_exact(ego_to_focus.len() - 1);

      log_trace!("process shortest path");

      for k in 0..ego_to_focus.len()-1 {
        let a = ego_to_focus[k];
        let b = ego_to_focus[k + 1];

        let a_kind = self.node_info_from_id(a)?.kind;
        let b_kind = self.node_info_from_id(b)?.kind;

        let a_b_weight = self.graph_from(context)?.get_edge(a, b).ok_or("Got invalid edge")?;

        if k + 2 == ego_to_focus.len() {
          if a_kind == NodeKind::User {
            edges.push((a, b, a_b_weight));
          } else {
            log_trace!("ignore node {}", self.node_info_from_id(a)?.name);
          }
        } else if b_kind != NodeKind::User {
          log_trace!("ignore node {}", self.node_info_from_id(b)?.name);
          let c = ego_to_focus[k + 2];
          let b_c_weight = self.graph_from(context)?.get_edge(b, c).ok_or("Got invalid edge")?;
          let a_c_weight = a_b_weight * b_c_weight * if a_b_weight < 0.0 && b_c_weight < 0.0 { -1.0 } else { 1.0 };
          edges.push((a, c, a_c_weight));
        } else if a_kind == NodeKind::User {
          edges.push((a, b, a_b_weight));
        } else {
          log_trace!("ignore node {}", self.node_info_from_id(a)?.name);
        }
      }

      log_trace!("add path to the graph");

      for (src, dst, weight) in edges {
        if !indices.contains_key(&src) {
          let index = im_graph.add_node(src);
          indices.insert(src, index);
          ids.insert(index, src);
        }

        if !indices.contains_key(&dst) {
          let index = im_graph.add_node(dst);
          indices.insert(dst, index);
          ids.insert(index, dst);
        }

        im_graph.add_edge(
          *indices.get(&src).ok_or("Got invalid node")?,
          *indices.get(&dst).ok_or("Got invalid node")?,
          weight
        );
      }
    }

    log_trace!("remove self references");

    for (_, src_index) in indices.iter() {
      let neighbors : Vec<_> =
        im_graph.edges(*src_index)
          .map(|edge| (edge.target(), edge.id()))
          .collect();

      for (dst_index, edge_id) in neighbors {
        if *src_index == dst_index {
          im_graph.remove_edge(edge_id);
        }
      }
    }

    let mut edge_ids = Vec::<(NodeId, NodeId, Weight)>::new();
    edge_ids.reserve_exact(indices.len() * 2); // ad hok

    log_trace!("build final array");

    for (_, src_index) in indices {
      for edge in im_graph.edges(src_index) {
        edge_ids.push((
          *ids.get(&src_index).ok_or("Got invalid node")?,
          *ids.get(&edge.target()).ok_or("Got invalid node")?,
          *edge.weight()
        ));
      }
    }

    let edge_names : Result<Vec<(&str, &str, Weight)>, BoxedError> =
      edge_ids
        .into_iter()
        .map(|(src_id, dst_id, weight)| {Ok((
          self.node_info_from_id(src_id)?.name.as_str(),
          self.node_info_from_id(dst_id)?.name.as_str(),
          weight
        ))})
        .collect();

    let result : Vec<_>  = edge_names?.into_iter().skip(index as usize).take(count as usize).collect();
    let v      : Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  pub fn read_connected(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_connected: `{}` `{}`", context, ego);

    let edges = self.connected_node_names(context, ego)?;

    if edges.is_empty() {
      return error!("read_connected", "No edges");
    }
    return Ok(rmp_serde::to_vec(&edges)?);
  }

  pub fn read_node_list(&self) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_node_list");

    let result : Vec<(&str,)> =
      self.node_infos
        .iter()
        .map(|info| (info.name.as_str(),))
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  pub fn read_edges(&mut self, context : &str) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD read_edges: `{}`", context);

    let infos = self.node_infos.clone();

    let mut v = Vec::<(&str, &str, Weight)>::new();
    v.reserve(infos.len() * 2); // ad hok

    for src_id in 0..infos.len() {
      let src_name = infos[src_id].name.as_str();

      for (dst_id, weight) in
        self
          .graph_from(context)?
          .neighbors_weighted(src_id, Neighbors::All)
          .unwrap_or(IntMap::default())
          .iter()
      {
        let dst_name =
          infos
            .get(*dst_id)
            .ok_or("Node does not exist")?.name
            .as_str();
        v.push((src_name, dst_name, *weight));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  pub fn read_mutual_scores(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<
    Vec<u8>,
    BoxedError
  > {
    log_info!("CMD read_mutual_scores: `{}` `{}`", context, ego);

    let ego_id = self.node_id_from_name(ego)?;

    let ranks = get_ranks_or_recalculate(self.graph_from(context)?, ego_id)?;

    let mut v = Vec::<(String, Weight, Weight)>::new();

    v.reserve_exact(ranks.len());

    for (node, score) in ranks {
      let info = self.node_info_from_id(node)?.clone();
      if score > 0.0 && info.kind == NodeKind::User
      {
        v.push((
          info.name,
          score,
          get_score_or_recalculate(self.graph_from(context)?, node, ego_id)?
        ));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  pub fn write_reset(&mut self) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_reset");

    self.reset()?;

    return Ok(rmp_serde::to_vec(&())?);
  }
}

//  ================================================
//
//    Zero node recalculation
//
//  ================================================

impl AugMultiGraph {
  fn reduced_graph(&mut self) -> Result<Vec<(NodeId, NodeId, f64)>, BoxedError> {
    log_trace!("reduced_graph");

    let zero = self.find_or_add_node_by_name("", ZERO_NODE.as_str())?;

    let users : Vec<NodeId> =
      self.node_infos
        .iter()
        .enumerate()
        .filter(|(id, info)|
          *id != zero && info.kind == NodeKind::User
        )
        .map(|(id, _)| id)
        .collect();

    if users.is_empty() {
      return Ok(Vec::new());
    }

    for id in users.iter() {
      self.graph_from("")?.calculate(*id, *NUM_WALK)?;
    }

    let edges : Vec<(NodeId, NodeId, Weight)> =
      users.into_iter()
        .map(|id| {
          let result : Result<_, BoxedError> =
            get_ranks_or_recalculate(self.graph_from("")?, id)?
              .into_iter()
              .map(|(node_id, score)| (id, node_id, score))
              .filter_map(|(ego_id, node_id, score)| {
                let kind = match self.node_info_from_id(node_id) {
                  Ok(info) => info.kind,
                  Err(x)   => return Some(error!("reduced_graph", "{}", x)),
                };
                if (kind == NodeKind::User || kind == NodeKind::Beacon) &&
                   score > 0.0 &&
                   ego_id != node_id
                {
                  Some(Ok((ego_id, node_id, score)))
                } else {
                  None
                }
              })
              .collect();
          Ok::<Vec<(NodeId, NodeId, Weight)>, BoxedError>(result?)
        })
        .filter_map(|res| res.ok())
        .flatten()
        .collect();

    let result : Result<Vec<(NodeId, NodeId, f64)>, BoxedError> =
      edges
        .into_iter()
        .map(|(ego_id, dst_id, weight)| -> Result<_, BoxedError> {
          let ego_kind = self.node_info_from_id(ego_id)?.kind;
          let dst_kind = self.node_info_from_id(dst_id)?.kind;
          Ok((ego_id, ego_kind, dst_id, dst_kind, weight))
        })
        .filter(|val| match val {
          Ok((ego_id, ego_kind, dst_id, dst_kind, _)) => {
            if *ego_id == zero || *dst_id == zero {
              false
            } else {
              return  ego_id != dst_id &&
                      *ego_kind == NodeKind::User &&
                     (*dst_kind == NodeKind::User || *dst_kind == NodeKind::Beacon);
            }
          },
          Err(_) => true,
        })
        .map(|val| { match val {
          Ok((ego_id, _, dst_id, _, weight)) => Ok((ego_id, dst_id, weight)),
          Err(x)                             => error!("reduced_graph", "{}", x),
        }})
        .collect();

    return Ok(result?);
  }

  fn delete_from_zero(&mut self) -> Result<(), BoxedError> {
    log_trace!("delete_from_zero");

    let src_id = self.node_id_from_name(ZERO_NODE.as_str())?;

    let edges = match self.connected_nodes("", src_id) {
      Ok(x) => x,
      _     => return Ok(()),
    };

    for (src, dst) in edges {
      self.set_edge("", src, dst, 0.0)?;
    }

    return Ok(());
  }

  fn top_nodes(&mut self) -> Result<Vec<(NodeId, f64)>, BoxedError> {
    log_trace!("top_nodes");

    let reduced = self.reduced_graph()?;

    if reduced.is_empty() {
      return error!("top_nodes", "Reduced graph empty");
    }

    let mut pr = Pagerank::<NodeId>::new();

    let zero = self.node_id_from_name(ZERO_NODE.as_str())?;

    reduced
      .iter()
      .filter(|(source, target, _weight)|
        *source != zero && *target != zero
      )
      .for_each(|(source, target, _weight)| {
        // TODO: check weight
        pr.add_edge(*source, *target);
      });

    log_verbose!("Calculate page rank");
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
      return error!("top_nodes", "No top nodes");
    }

    return Ok(res);
  }

  pub fn write_recalculate_zero(&mut self) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_recalculate_zero");

    let _ = self.find_or_add_node_by_name("", ZERO_NODE.as_str())?;

    self.recalculate_all(0)?; // FIXME Ad hok hack
    self.delete_from_zero()?;

    let nodes = self.top_nodes()?;

    self.recalculate_all(0)?; // FIXME Ad hok hack
    {
      let zero = self.node_id_from_name(ZERO_NODE.as_str())?;

      for (node_id, amount) in nodes.iter() {
        self.set_edge("", zero, *node_id, *amount)?;
      }
    }
    self.recalculate_all(*NUM_WALK)?; // FIXME Ad hok hack

    return Ok(rmp_serde::to_vec(&())?);
  }
}

//  ================================================
//
//    The service
//
//  ================================================

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

    let mut graph = match data.graph_writable.lock() {
      Ok(x)  => x,
      Err(e) => return error!("perform_command", "{}", e),
    };
    let mut res : Option<_> = None;
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
      Ok(ref mut x) => x.copy_from(graph.deref_mut()),
      Err(e) => {
        return error!("perform_command", "{}", e);
      },
    };
    if let Some(x) = res {
      return x;
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
      CMD_VERSION => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          return read_version();
        }
      },
      CMD_LOG_LEVEL => {
        if let Ok(log_level) = rmp_serde::from_slice(command.payload.as_slice()) {
          return write_log_level(log_level);
        }
      },
      CMD_SYNC => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          let mut queue = data.queue_commands.lock().expect("Mutex lock failed");
          while !queue.is_empty() {
            queue = data.cond_done.wait(queue).expect("Condvar wait failed");
          }
          return Ok(rmp_serde::to_vec(&())?);
        }
      },
      //  Read commands
      CMD_NODE_LIST => {
        if let Ok(()) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_node_list();
        }
      },
      CMD_NODE_SCORE_NULL => {
        if let Ok((ego, target)) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_node_score_null(ego, target);
        }
      },
      CMD_SCORES_NULL => {
        if let Ok(ego) = rmp_serde::from_slice(command.payload.as_slice()) {
          return graph.read_scores_null(ego);
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
    for cmd in queue.iter() {
      match perform_command(&data, cmd.clone()) {
        Ok(_)  => {},
        Err(e) => {
          log_error!("(write_queue_thread) {}", e);
        },
      };
    }
    queue.clear();
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
  data.cond_add.notify_one();
}

fn decode_and_handle_request(
  data    : Data,
  request : &[u8]
) -> Result<Vec<u8>, BoxedError> {
  log_trace!("decode_and_handle_request");

  let command : Command;

  match rmp_serde::from_slice(request) {
    Ok((command_value, context_value, payload_value)) => {
      command = Command {
        id      : command_value,
        context : context_value,
        payload : payload_value,
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
    command.id == CMD_NODE_SCORE_NULL  ||
    command.id == CMD_SCORES_NULL      ||
    command.id == CMD_NODE_LIST
  ) {
    return error!("decode_and_handle_request", "Context should be empty");
  }

  if command.id == CMD_RESET            ||
     command.id == CMD_RECALCULATE_ZERO ||
     command.id == CMD_DELETE_EDGE      ||
     command.id == CMD_DELETE_NODE      ||
     command.id == CMD_PUT_EDGE
  {
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
