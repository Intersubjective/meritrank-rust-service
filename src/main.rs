//  ================================================================
//
//    Modules and dependencies
//
//  ================================================================

mod astar;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ops::DerefMut;
use itertools::Itertools;
use std::env::var;
use std::string::ToString;
use std::error::Error;
use petgraph::visit::EdgeRef;
use petgraph::graph::{DiGraph, NodeIndex};
use nng::{Aio, AioResult, Context, Message, Protocol, Socket};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, IntMap, Weight, NodeId, Neighbors};
use ctrlc;
use chrono;
use crate::astar::astar::*;

//  ================================================================
//
//    Global options
//
//  ================================================================

pub static ERROR   : bool = true;
pub static WARNING : bool = true;
pub static INFO    : bool = true;
pub static VERBOSE : bool = true;
pub static TRACE   : bool = true;

const VERSION : &str = match option_env!("CARGO_PKG_VERSION") {
  Some(x) => x,
  None    => "dev",
};

lazy_static::lazy_static! {
  static ref SERVICE_URL : String =
    var("MERITRANK_SERVICE_URL")
      .unwrap_or("tcp://127.0.0.1:10234".to_string());

  static ref THREADS : usize =
    var("MERITRANK_SERVICE_THREADS")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(1);

  static ref NUM_WALK : usize =
    var("MERITRANK_NUM_WALK")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(10000);

  static ref ZERO_NODE : String =
    var("MERITRANK_ZERO_NODE")
      .unwrap_or("U000000000000".to_string());

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
  match LOG_MUTEX.lock() {
    Ok(_) => {
      println!("{:?}  {}{}", chrono::offset::Local::now(), prefix, message);
    },
    _ => {
      println!("{:?}  LOG MUTEX FAILED", chrono::offset::Local::now());
    },
  };
}


macro_rules! log_error {
  ($($arg:expr),*) => {
    if ERROR {
      log_with_time("ERROR   ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_warning {
  ($($arg:expr),*) => {
    if WARNING {
      log_with_time("WARNING ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_info {
  ($($arg:expr),*) => {
    if INFO {
      log_with_time("INFO    ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_verbose {
  ($($arg:expr),*) => {
    if VERBOSE {
      log_with_time("VERBOSE --- ", format!($($arg),*).as_str());
    }
  };
}

macro_rules! log_trace {
  ($($arg:expr),*) => {
    if TRACE {
      log_with_time("TRACE   --- --- ", format!($($arg),*).as_str());
    }
  };
}

fn kind_from_name(name : &str) -> NodeKind {
  log_verbose!("kind_from_name");

  match name.chars().nth(0) {
    Some('U') => NodeKind::User,
    Some('B') => NodeKind::Beacon,
    Some('C') => NodeKind::Comment,
    _         => NodeKind::Unknown,
  }
}

impl AugMultiGraph {
  pub fn new() -> Result<AugMultiGraph, Box<dyn Error + 'static>> {
    log_verbose!("AugMultiGraph::new");

    Ok(AugMultiGraph {
      node_count   : 0,
      node_infos   : Vec::new(),
      node_ids     : HashMap::new(),
      contexts     : HashMap::new(),
    })
  }

  fn reset(&mut self) -> Result<(), Box<dyn Error + 'static>> {
    log_verbose!("reset");

    self.node_count   = 0;
    self.node_infos   = Vec::new();
    self.node_ids     = HashMap::new();
    self.contexts     = HashMap::new();

    Ok(())
  }

  fn node_id_from_name(&self, node_name : &str) -> Result<NodeId, Box<dyn Error + 'static>> {
    log_verbose!("node_id_from_name");

    match self.node_ids.get(node_name) {
      Some(x) => Ok(*x),
      _       => {
        log_error!("(AugMultiGraph::node_id_from_name) Node does not exist: `{}`", node_name);
        Err("Node does not exist".into())
      },
    }
  }

  fn node_info_from_id(&self, node_id : NodeId) -> Result<&NodeInfo, Box<dyn Error + 'static>> {
    log_verbose!("node_info_from_id");

    match self.node_infos.get(node_id) {
      Some(x) => Ok(x),
      _       => {
        log_error!("(AugMultiGraph::node_info_from_id) Node does not exist: `{}`", node_id);
        Err("Node does not exist".into())
      },
    }
  }

  fn find_or_add_node_by_name(
    &mut self,
    context   : &str,
    node_name : &str
  ) -> Result<NodeId, Box<dyn Error + 'static>> {
    log_verbose!("find_or_add_node_by_name");

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

        log_trace!("add node in NULL: {}", node_id);
        self.graph_from("")?.add_node(node_id, ());
      }

      self.add_context_if_does_not_exist(context)?;

      log_trace!("add node in {}: {}", context, node_id);
      self.graph_from(context)?.add_node(node_id, ());

      Ok(node_id)
    }
  }

  fn add_context_if_does_not_exist(&mut self, context : &str) -> Result<(), Box<dyn Error + 'static>> {
    log_verbose!("add_context_if_does_not_exist");

    if !self.contexts.contains_key(context) {
      self.contexts.insert(context.to_string(), MeritRank::new(Graph::new())?);
    }
    Ok(())
  }

  //  Get mutable graph from a context
  //
  fn graph_from(&mut self, context : &str) -> Result<&mut MeritRank<()>, Box<dyn Error + 'static>> {
    log_verbose!("graph_from");

    match self.contexts.get_mut(context) {
      Some(graph) => Ok(graph),
      None        => {
        log_error!("(graph_from) Context does not exist: `{}`", context);
        Err("Context does not exist".into())
      },
    }
  }

  fn set_edge(
    &mut self,
    context : &str,
    src     : NodeId,
    dst     : NodeId,
    amount  : f64
  ) -> Result<(), Box<dyn Error + 'static>> {
    log_verbose!("set_edge");

    if context.is_empty() {
      self.add_context_if_does_not_exist("")?;
      log_trace!("add edge in NULL: {} -> {} for {}", src, dst, amount);
      self.graph_from("")?.add_edge(src, dst, amount);
    } else {
      self.add_context_if_does_not_exist("")?;
      self.add_context_if_does_not_exist(context)?;

      //  This doesn't make any sense but it's Rust.

      let null_weight = self.graph_from("")?     .get_edge(src, dst).unwrap_or(0.0);
      let old_weight  = self.graph_from(context)?.get_edge(src, dst).unwrap_or(0.0);
      let delta       = null_weight + amount - old_weight;

      log_trace!("add edge in NULL: {} -> {} for {}", src, dst, delta);
      self.graph_from("")?.add_edge(src, dst, delta);

      log_trace!("add edge in {}: {} -> {} for {}", context, src, dst, amount);
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
    Box<dyn Error + 'static>
  > {
    log_verbose!("connected_nodes");

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
    Box<dyn Error + 'static>
  > {
    log_verbose!("connected_node_names");

    let src_id = self.node_id_from_name(ego)?;

    let edge_ids = self.connected_nodes(context, src_id)?;

    let res : Result<Vec<(&str, &str)>, Box<dyn Error + 'static>> =
      edge_ids
        .into_iter()
        .map(|(src_id, dst_id)| Ok((
          self.node_info_from_id(src_id)?.name.as_str(),
          self.node_info_from_id(dst_id)?.name.as_str()
        )))
        .collect();

    Ok(res?)
  }

  fn recalculate_all(&mut self, num_walk : usize) -> Result<(), Box<dyn Error + 'static>> {
    log_verbose!("recalculate_all");

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

fn read_version() -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  log_info!("CMD read_version");

  let s : String = VERSION.to_string();
  Ok(rmp_serde::to_vec(&s)?)
}

impl AugMultiGraph {
  fn read_node_score_null(&mut self, ego : &str, target : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_node_score_null");

    let ego_id    = self.node_id_from_name(ego)?;
    let target_id = self.node_id_from_name(target)?;

    let mut w : Weight = 0.0;
    for (_, rank) in self.contexts.iter_mut() {
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

  fn read_scores_null(&mut self, ego : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_scores_null");

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

    let result : Result<Vec<(&str, &str, Weight)>, _> =
      intermediate
        .iter()
        .map(|(ego, node, weight)|
          match self.node_info_from_id(*node) {
            Ok(info) => Ok((*ego, info.name.as_str(), *weight)),
            Err(x)   => Err(x)
          }
        )
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result?)?;
    Ok(v)
  }

  fn read_node_score(
    &mut self,
    context : &str,
    ego     : &str,
    target  : &str
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_node_score");

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

  fn read_scores(
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
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_scores");

    let kind = match kind_str {
      ""  => NodeKind::Unknown,
      "U" => NodeKind::User,
      "B" => NodeKind::Beacon,
      "C" => NodeKind::Comment,
       _  => {
         log_error!("Invalid node kind string: {}", kind_str);
         return Err(format!("Invalid node kind \"{}\"; only \"U\", \"B\", \"C\" are allowed", kind_str).into());
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

    let im1 : Result<Vec<(NodeId, NodeKind, Weight)>, Box<dyn Error + 'static>> =
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

    let im2 : Result<Vec<(NodeId, NodeKind, Weight)>, Box<dyn Error + 'static>> =
      im1?
        .into_iter()
        .map(|x| Ok(x))
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
          Err(x) => Some(Err(x)),
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
        .map(|(target_id, _, weight)| {
          match self.node_info_from_id(*target_id) {
            Ok(x)  => Ok((ego, x.name.as_str(), *weight)),
            Err(x) => Err(x)
          }
        })
        .collect();

    Ok(rmp_serde::to_vec(&page?)?)
  }

  fn write_put_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
    amount  : f64
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD write_put_edge");

    let src_id = self.find_or_add_node_by_name(context, src)?;
    let dst_id = self.find_or_add_node_by_name(context, dst)?;

    self.set_edge(context, src_id, dst_id, amount)?;

    let result: Vec<(&str, &str, f64)> = [(src, dst, amount)].to_vec();
    Ok(rmp_serde::to_vec(&result)?)
  }

  fn write_delete_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD write_delete_edge");

    let src_id = self.node_id_from_name(src)?;
    let dst_id = self.node_id_from_name(dst)?;

    self.set_edge(context, src_id, dst_id, 0.0)?;

    Ok(EMPTY_RESULT.to_vec())
  }

  fn write_delete_node(
    &mut self,
    context : &str,
    node    : &str,
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD write_delete_node");

    let id = self.node_id_from_name(node)?;

    for n in self.graph_from(context)?.neighbors_weighted(id, Neighbors::All).ok_or("Unable to get neighbors")?.keys() {
      self.set_edge(context, id, *n, 0.0)?;
    }

    Ok(EMPTY_RESULT.to_vec())
  }

  fn read_graph(
    &mut self,
    context       : &str,
    ego           : &str,
    focus         : &str,
    positive_only : bool,
    index         : u32,
    count         : u32
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_graph");

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

      let neighbor = |node : NodeId, index : usize| -> Result<Option<Link<NodeId, Weight>>, Box<dyn Error + 'static>> {
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

      let heuristic = |node : NodeId| -> Result<Weight, Box<dyn Error + 'static>> {
        //  ad hok
        Ok(((node as i64) - (focus_id as i64)).abs() as f64)
      };

      let mut astar_state = init(ego_id, focus_id, Weight::MAX);

      let mut status = Status::PROGRESS;
      let mut count  = 0;

      //  Do 10000 iterations max
      //

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

    let edge_names : Result<Vec<(&str, &str, Weight)>, Box<dyn Error + 'static>> =
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

  fn read_connected(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_connected");

    let edges = self.connected_node_names(context, ego)?;

    if edges.is_empty() {
      return Err("No edges".into());
    }
    return Ok(rmp_serde::to_vec(&edges)?);
  }

  fn read_nodelist(&self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_nodelist");

    let result : Vec<(&str,)> =
      self.node_infos
        .iter()
        .map(|info| (info.name.as_str(),))
        .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
  }

  fn read_edges(&mut self, context : &str) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD read_edges");

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

  fn read_mutual_scores(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<
    Vec<u8>,
    Box<dyn Error + 'static>
  > {
    log_info!("CMD read_mutual_scores");

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

  fn write_reset(&mut self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
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
  fn reduced_graph(&mut self) -> Result<Vec<(NodeId, NodeId, f64)>, Box<dyn Error + 'static>> {
    log_verbose!("reduced_graph");

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
          let result : Result<Vec<(NodeId, NodeId, Weight)>, _> =
            self
              .graph_from("")?
              .get_ranks(id, None)?
              .into_iter()
              .map(|(node_id, score)| (id, node_id, score))
              .filter_map(|(ego_id, node_id, score)| {
                let kind = match self.node_info_from_id(node_id) {
                  Ok(info) => info.kind,
                  Err(x)   => return Some(Err(x)),
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
          Ok::<Vec<(NodeId, NodeId, Weight)>, Box<dyn Error + 'static>>(result?)
        })
        .filter_map(|res| res.ok())
        .flatten()
        .collect();

    let result : Result<Vec<(NodeId, NodeId, f64)>, Box<dyn Error + 'static>> =
      edges
        .into_iter()
        .map(|(ego_id, dst_id, weight)| {
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
        .map(|val| match val {
          Ok((ego_id, _, dst_id, _, weight)) => Ok((ego_id, dst_id, weight)),
          Err(x)                             => Err(x),
        })
        .collect();

    return Ok(result?);
  }

  fn delete_from_zero(&mut self) -> Result<(), Box<dyn Error + 'static>> {
    log_verbose!("delete_from_zero");

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

  fn top_nodes(&mut self) -> Result<Vec<(NodeId, f64)>, Box<dyn Error + 'static>> {
    log_verbose!("top_nodes");

    let reduced = self.reduced_graph()?;

    if reduced.is_empty() {
      return Err("Reduced graph empty".into());
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

  fn write_zerorec(&mut self) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
    log_info!("CMD write_zerorec");

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
) -> Result<Vec<u8>, Box<dyn Error + 'static>> {
  log_verbose!("decode_and_handle_request");

  let mut context : &str  = "";
  let mut payload : &[u8] = request;

  if let Ok(("context", context_value, contexted_payload)) = rmp_serde::from_slice(payload) {
    context = context_value;
    payload = contexted_payload;
  }

  if let Ok("ver") = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      read_version()
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else if let Ok(((("src", "=", ego), ("dest", "=", target)), (), "null")) = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      multi_graph.read_node_score_null(ego, target)
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else if let Ok(((("src", "=", ego), ), (), "null")) = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      multi_graph.read_scores_null(ego)
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else if let Ok(((("src", "=", ego), ("dest", "=", target)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_node_score(context, ego, target)
  } else if let Ok(((("src", "=", ego), ), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, "",        false,         f64::MAX, true,  f64::MIN, true,  0,     u32::MAX)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, false,         score_lt, false, score_gt, false, 0,     u32::MAX)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">=", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, false,         score_lt, false, score_gt, true,  0,     u32::MAX)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, false,         score_lt, true,  score_gt, false, 0,     u32::MAX)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("score", ">=", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, false,         score_lt, true,  score_gt, true,  0,     u32::MAX)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, hide_personal, score_lt, false, score_gt, false, index, count)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">=", score_gt), ("score", "<", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, hide_personal, score_lt, false, score_gt, true,  index, count)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<=", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, hide_personal, score_lt, true,  score_gt, false, index, count)
  } else if let Ok(((("src", "=", ego), ("node_kind", node_kind), ("hide_personal", hide_personal), ("score", ">=", score_gt), ("score", "<=", score_lt), ("index", index), ("count", count)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_scores(context, ego, node_kind, hide_personal, score_lt, true,  score_gt, true,  index, count)
  } else if let Ok((((subject, object, amount), ), ())) = rmp_serde::from_slice(payload) {
    multi_graph.write_put_edge(context, subject, object, amount)
  } else if let Ok(((("src", "delete", ego), ("dest", "delete", target)), ())) = rmp_serde::from_slice(payload) {
    multi_graph.write_delete_edge(context, ego, target)
  } else if let Ok(((("src", "delete", ego), ), ())) = rmp_serde::from_slice(payload) {
    multi_graph.write_delete_node(context, ego)
  } else if let Ok((((ego, "gravity", focus), positive_only, index, count), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_graph(context, ego, focus, positive_only, index, count)
  } else if let Ok((((ego, "connected"), ), ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_connected(context, ego)
  } else if let Ok(("nodes", ())) = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      multi_graph.read_nodelist()
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else if let Ok(("edges", ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_edges(context)
  } else if let Ok(("users_stats", ego, ())) = rmp_serde::from_slice(payload) {
    multi_graph.read_mutual_scores(context, ego)
  } else if let Ok(("reset", ())) = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      multi_graph.write_reset()
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else if let Ok(("zerorec", ())) = rmp_serde::from_slice(payload) {
    if context.is_empty() {
      multi_graph.write_zerorec()
    } else {
      Err(format!("Invalid request: {:?}", request).into())
    }
  } else {
    Err(format!("Invalid request: {:?}", request).into())
  }
}

fn process(
  multi_graph : &mut AugMultiGraph,
  req         : Message
) -> Vec<u8> {
  log_verbose!("process");

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
  log_verbose!("worker_callback");

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

fn main_sync() -> Result<(), Box<dyn Error + 'static>> {
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

fn main_async(threads : usize) -> Result<(), Box<dyn Error + 'static>> {
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
