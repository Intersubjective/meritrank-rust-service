use std::{
  sync::atomic::{AtomicBool, Ordering},
  collections::HashMap,
  sync::{Arc, Mutex},
  ops::DerefMut,
  env::var,
  string::ToString,
  error::Error,
  thread
};
use chrono;
use itertools::Itertools;
use nng::{Aio, AioResult, Context, Message, Protocol, Socket};
use petgraph::{visit::EdgeRef, graph::{DiGraph, NodeIndex}};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, IntMap, NodeId, Neighbors};

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

  static ref EMPTY_RESULT : Vec<u8> = {
    const EMPTY_ROWS_VEC : Vec<(&str, &str, f64)> = Vec::new();
    rmp_serde::to_vec(&EMPTY_ROWS_VEC).expect("Unable to serialize empty result")
  };
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
pub struct AugMultiGraph {
  pub node_count : usize,
  pub node_infos : Vec<NodeInfo>,
  pub node_ids   : HashMap<String, NodeId>,
  pub contexts   : HashMap<String, MeritRank<()>>,
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

  fn add_context_if_does_not_exist(&mut self, context : &str) -> Result<(), BoxedError> {
    log_trace!("add_context_if_does_not_exist: `{}`", context);

    if !self.contexts.contains_key(context) {
      if context.is_empty() {
        log_verbose!("Add context: NULL");
      } else {
        log_verbose!("Add context: `{}`", context);
      }
      self.contexts.insert(context.to_string(), MeritRank::new(Graph::new())?);
    }
    Ok(())
  }

  //  Get mutable graph from a context
  //
  fn graph_from(&mut self, context : &str) -> Result<&mut MeritRank<()>, BoxedError> {
    log_trace!("graph_from: `{}`", context);

    match self.contexts.get_mut(context) {
      Some(graph) => Ok(graph),
      None        => {
        error!("graph_from", "Context does not exist: `{}`", context)
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
        self.add_context_if_does_not_exist("")?;

        log_verbose!("Add node in NULL: {}", node_id);
        self.graph_from("")?.add_node(node_id, ());
      }

      self.add_context_if_does_not_exist(context)?;

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
      self.add_context_if_does_not_exist("")?;
      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, amount);
      self.graph_from("")?.add_edge(src, dst, amount);
    } else {
      self.add_context_if_does_not_exist("")?;
      self.add_context_if_does_not_exist(context)?;

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

    let w = match graph.get_node_score(ego_id, target_id) {
      Ok(x) => x,
      _     => {
        log_warning!("(read_node_score) Node scores recalculation for {}", ego_id);
        graph.calculate(ego_id, *NUM_WALK)?;
        graph.get_node_score(ego_id, target_id)?
      }
    };

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

    let ranks = match self.graph_from(context)?.get_ranks(node_id, None) {
      Ok(x) => x,
      _     => {
        log_warning!("(read_scores) Node scores recalculation for {}", node_id);
        let graph = self.graph_from(context)?;
        let _ = graph.calculate(node_id, *NUM_WALK)?;
        graph.get_ranks(node_id, None)?
      }
    };

    //  FIXME
    //  Fix this boilerplate madness!!!
    //
    //  Problems:
    //  1.  We have to use ? operator inside of closures, so we can handle errors properly.
    //  2.  We have to borrow self multiple times at once to call methods.

    let im1 : Result<Vec<(NodeId, NodeKind, Weight)>, BoxedError> =
      ranks
        .into_iter()
        .map(|(n, w)| {
            Ok((
              n,
              self.node_info_from_id(n)?.kind,
              w,
            ))
        })
        .filter(|val|
          match val {
            Ok((_, NodeKind::Unknown, _)) => true,
            Ok((_, target_kind, _))       => kind == *target_kind,
            _                             => true,
          }
        )
        .filter(|val| match val {
          Ok((_, _, score)) => score_gt < *score || (score_gte && score_gt == *score),
          _                 => true,
        })
        .filter(|val| match val {
          Ok((_, _, score)) => *score < score_lt || (score_lte && score_lt == *score),
          _                 => true,
        })
        .collect();

    let im2 : Result<Vec<(NodeId, NodeKind, Weight)>, BoxedError> =
      im1?
        .into_iter()
        .map(|x| -> Result<_, BoxedError> { Ok(x) })
        .filter_map(|val| match val {
          Ok((target_id, target_kind, weight)) =>
            if hide_personal {
              if target_kind == NodeKind::Comment || target_kind == NodeKind::Beacon {
                match self.graph_from(context) {
                  Ok(graph) =>
                    if graph.get_edge(target_id, node_id).is_some() {
                      None
                    } else {
                      Some(Ok((target_id, target_kind, weight)))
                    },
                  _ => Some(Ok((target_id, target_kind, weight))),
                }
              } else {
                Some(Ok((target_id, target_kind, weight)))
              }
            } else {
              Some(Ok((target_id, target_kind, weight)))
            }
          Err(x) => Some(error!("read_scores", "{}", x)),
        })
        .collect();

    //
    //  ================================
    //

    let mut sorted = im2?;
    sorted.sort_by(|(_, _, a), (_, _, b)| a.total_cmp(b));

    let page : Result<Vec<(&str, &str, Weight)>, _> =
      sorted
        .iter()
        .skip(index as usize)
        .take(count as usize)
        .map(|(target_id, _, weight)| -> Result<_, BoxedError> {
          match self.node_info_from_id(*target_id) {
            Ok(x)  => Ok((ego, x.name.as_str(), *weight)),
            Err(x) => error!("read_scores", "{}", x)
          }
        })
        .collect();

    Ok(rmp_serde::to_vec(&page?)?)
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

    let result: Vec<(&str, &str, f64)> = [(src, dst, amount)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
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

    Ok(EMPTY_RESULT.to_vec())
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

    Ok(EMPTY_RESULT.to_vec())
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

    let focus_neighbors = self.graph_from(context)?.neighbors_weighted(focus_id, Neighbors::All).ok_or("Focus has no neighbors")?;

    let mut indices  = HashMap::<NodeId, NodeIndex>::new();
    let mut ids      = HashMap::<NodeIndex, NodeId>::new();
    let mut im_graph = DiGraph::<NodeId, Weight>::new();

    {
      let index = im_graph.add_node(focus_id);
      indices.insert(focus_id, index);
      ids.insert(index, focus_id);
    }

    log_trace!("enumerate focus neighbors");

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
        im_graph.add_edge(*indices.get(&focus_id).ok_or("Got invalid node")?, *indices.get(&dst_id).ok_or("Got invalid node")?, focus_dst_weight);
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
          im_graph.add_edge(*indices.get(&focus_id).ok_or("Got invalid node")?, *indices.get(&ngh_id).ok_or("Got invalid node")?, focus_ngh_weight);
        }
      }
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

      let heuristic = |node : NodeId| -> Result<Weight, BoxedError> {
        //  ad hok
        Ok(((node as i64) - (focus_id as i64)).abs() as f64)
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
          }
        } else if b_kind != NodeKind::User {
          let c = ego_to_focus[k + 2];
          let b_c_weight = self.graph_from(context)?.get_edge(b, c).ok_or("Got invalid edge")?;
          let a_c_weight = a_b_weight * b_c_weight * if a_b_weight < 0.0 && b_c_weight < 0.0 { -1.0 } else { 1.0 };
          edges.push((a, c, a_c_weight));
        } else if a_kind == NodeKind::User {
          edges.push((a, b, a_b_weight));
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
        im_graph.add_edge(*indices.get(&src).ok_or("Got invalid node")?, *indices.get(&dst).ok_or("Got invalid node")?, weight);
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

    let ranks = match self.graph_from(context)?.get_ranks(ego_id, None) {
      Ok(x) => x,
      _     => {
        log_warning!("(read_mutual_scores) Node scores recalculation for {}", ego_id);
        let graph = self.graph_from(context)?;
        graph.calculate(ego_id, *NUM_WALK)?;
        graph.get_ranks(ego_id, None)?
      }
    };

    let mut v = Vec::<(String, Weight, Weight)>::new();

    v.reserve_exact(ranks.len());

    for (node, score) in ranks {
      let info = self.node_info_from_id(node)?.clone();
      if score > 0.0 && info.kind == NodeKind::User
      {
        let graph = self.graph_from(context)?;
        v.push((
          info.name,
          score,
          match graph.get_node_score(node, ego_id) {
            Ok(x) => x,
            _     => {
              log_warning!("(read_mutual_scores) Node scores recalculation for {}", ego_id);
              graph.calculate(node, *NUM_WALK)?;
              graph.get_node_score(node, ego_id)?
            }
          }
        ));
      }
    }

    Ok(rmp_serde::to_vec(&v)?)
  }

  pub fn write_reset(&mut self) -> Result<Vec<u8>, BoxedError> {
    log_info!("CMD write_reset");

    self.reset()?;

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
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
            self
              .graph_from("")?
              .get_ranks(id, None)?
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

    let _ = self.add_context_if_does_not_exist("");
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

    return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
  }
}

//  ================================================
//
//    The service
//
//  ================================================

fn decode_and_handle_request(
  multi_graph : &mut AugMultiGraph,
  request     : &[u8]
) -> Result<Vec<u8>, BoxedError> {
  log_trace!("decode_and_handle_request");

  let command : &str;
  let context : &str;
  let payload : Vec<u8>;

  match rmp_serde::from_slice(request) {
    Ok((command_value, context_value, payload_value)) => {    
      command = command_value;
      context = context_value;
      payload = payload_value;

      if context.is_empty() {
        log_trace!("decoded command `{}` in NULL with payload {:?}", command, payload);
      } else {
        log_trace!("decoded command `{}` in `{}` with payload {:?}", command, context, payload);
      }
    },

    Err(error) =>
      return error!("decode_and_handle_request", "Invalid request: {:?}; decoding error: {}", request, error),
  }

  if !context.is_empty() && (command == CMD_VERSION || command == CMD_LOG_LEVEL || command == CMD_RESET || command == CMD_RECALCULATE_ZERO || command == CMD_NODE_SCORE_NULL || command == CMD_SCORES_NULL || command == CMD_NODE_LIST) {
    return error!("decode_and_handle_request", "Context should be empty");
  }

  if        command == CMD_VERSION {
    if let Ok(()) = rmp_serde::from_slice(payload.as_slice()) {
      return read_version();
    }
  } else if command == CMD_LOG_LEVEL {
    if let Ok(log_level) = rmp_serde::from_slice(payload.as_slice()) {
      return write_log_level(log_level);
    }
  } else if command == CMD_RESET {
    if let Ok(()) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.write_reset();
    }
  } else if command == CMD_RECALCULATE_ZERO {  
    if let Ok(()) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.write_recalculate_zero();
    }
  } else if command == CMD_NODE_LIST {
    if let Ok(()) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_node_list();
    }
  } else if command == CMD_NODE_SCORE_NULL {
    if let Ok((ego, target)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_node_score_null(ego, target);
    }
  } else if command == CMD_SCORES_NULL {
    if let Ok(ego) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_scores_null(ego);
    }
  } else if command == CMD_NODE_SCORE {
    if let Ok((ego, target)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_node_score(context, ego, target);
    }
  } else if command == CMD_SCORES {
    if let Ok((ego, kind, hide_personal, lt, lte, gt, gte, index, count)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_scores(context, ego, kind, hide_personal, lt, lte, gt, gte, index, count);
    }
  } else if command == CMD_PUT_EDGE {
    if let Ok((src, dst, amount)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.write_put_edge(context, src, dst, amount);
    }
  } else if command == CMD_DELETE_EDGE {
    if let Ok((src, dst)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.write_delete_edge(context, src, dst);
    }
  } else if command == CMD_DELETE_NODE {
    if let Ok(node) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.write_delete_node(context, node);
    }
  } else if command == CMD_GRAPH {
    if let Ok((ego, focus, positive_only, index, count)) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_graph(context, ego, focus, positive_only, index, count);
    }
  } else if command == CMD_CONNECTED {
    if let Ok(node) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_connected(context, node);
    }
  } else if command == CMD_EDGES {
    if let Ok(()) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_edges(context);
    }
  } else if command == CMD_MUTUAL_SCORES {
    if let Ok(ego) = rmp_serde::from_slice(payload.as_slice()) {
      return multi_graph.read_mutual_scores(context, ego);
    }
  } else {
    return error!("decode_and_handle_request", "Unknown command: `{}`", command);
  }

  return error!("decode_and_handle_request", "Invalid payload for command `{}`: {:?}", command, payload);
}

fn process(
  multi_graph : &mut AugMultiGraph,
  req         : Message
) -> Vec<u8> {
  log_trace!("process");

  match decode_and_handle_request(multi_graph, req.as_slice()) {
    Ok(bytes)  => bytes,
    Err(error) => match rmp_serde::to_vec(&error.to_string()) {
      Ok(bytes)  => bytes,
      Err(error) => {
        log_error!("(process) Unable to serialize error: {:?}", error);
        Vec::new()
      },
    },
  }
}

fn worker_callback(multi_graph : Arc<Mutex<AugMultiGraph>>, aio : Aio, ctx : &Context, res : AioResult) {
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
      match multi_graph.lock() {
        Ok(ref mut guard) => {
          let msg : Vec<u8> = process(
            guard.deref_mut(),
            req
          );
          match ctx.send(&aio, msg.as_slice()) {
            Ok(_) => {},
            Err(error) => {
              log_error!("(worker_callback) SEND failed: {:?}", error);
            }
          };
        },
        Err(error) => {
          log_error!("(worker_callback) Mutex lock failed: {:?}", error);
        },
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

pub fn main_sync() -> Result<(), BoxedError> {
  log_info!("Starting server {} at {}", VERSION, *SERVICE_URL);
  log_info!("NUM_WALK={}", *NUM_WALK);

  let mut multi_graph = AugMultiGraph::new()?;

  let s = Socket::new(Protocol::Rep0)?;

  s.listen(&SERVICE_URL)?;

  loop {
    let request : Message = s.recv()?;
    let reply   : Vec<u8> = process(&mut multi_graph, request);
    let _ = s.send(reply.as_slice()).map_err(|(_, e)| e)?;
  }
}

pub fn main_async(threads : usize) -> Result<(), BoxedError> {
  log_info!("Starting server {} at {}, {} threads", VERSION, *SERVICE_URL, threads);
  log_info!("NUM_WALK={}", *NUM_WALK);

  let multi_graph = Arc::<Mutex<AugMultiGraph>>::new(Mutex::<AugMultiGraph>::new(AugMultiGraph::new()?));

  let s = Socket::new(Protocol::Rep0)?;

  let workers : Vec<_> = (0..threads)
    .map(|_| {
      let ctx                = Context::new(&s)?;
      let ctx_cloned         = ctx.clone();
      let multi_graph_cloned = multi_graph.clone();

      let aio = Aio::new(move |aio, res| {
        worker_callback(
          multi_graph_cloned.clone(),
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
