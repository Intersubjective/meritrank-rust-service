use std::{
  sync::atomic::Ordering,
  collections::HashMap,
  env::var,
  string::ToString,
  error::Error
};
use petgraph::{visit::EdgeRef, graph::{DiGraph, NodeIndex}};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, NodeId, Neighbors, MeritRankError};

use crate::log_error;
use crate::log_warning;
use crate::log_info;
use crate::log_verbose;
use crate::log_trace;
use crate::error;
use crate::log::*;
use crate::astar::astar::*;

pub use meritrank::Weight;

//  ================================================================
//
//    Constants
//
//  ================================================================

pub const VERSION : &str = match option_env!("CARGO_PKG_VERSION") {
  Some(x) => x,
  None    => "dev",
};

lazy_static::lazy_static! {
  pub static ref ZERO_NODE : String =
    var("MERITRANK_ZERO_NODE")
      .unwrap_or("U000000000000".to_string());

  pub static ref NUM_WALK : usize =
    var("MERITRANK_NUM_WALK")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(10000);

  pub static ref TOP_NODES_LIMIT : usize =
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

pub type BoxedError = Box<dyn Error + 'static>;

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
  pub contexts   : HashMap<String, MeritRank>,
}

//  ================================================================
//
//    Utils
//
//  ================================================================

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
  graph   : &mut MeritRank,
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
  graph   : &mut MeritRank,
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

fn all_neighbors(meritrank : &mut MeritRank, id : NodeId) -> Vec<(NodeId, Weight)> {
  let mut v = vec![];

  match meritrank.graph.get_node_data(id) {
    None => {},
    Some(data) => {
      v.reserve_exact(
        data.neighbors(Neighbors::Positive).len() +
        data.neighbors(Neighbors::Negative).len()
      );

      for x in data.neighbors(Neighbors::Positive) {
        v.push((*x.0, *x.1));
      }

      for x in data.neighbors(Neighbors::Negative) {
        v.push((*x.0, *x.1));
      }
    }
  }

  v
}

fn edge_weight_or_zero(
  meritrank : &mut MeritRank,
  src       : NodeId,
  dst       : NodeId
) -> Weight {
  match meritrank.graph.edge_weight(src, dst) {
    Ok(Some(x)) => *x,
    _           => 0.0,
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

  pub fn copy_from(&mut self, other : &AugMultiGraph) {
    self.node_count = other.node_count;
    self.node_infos = other.node_infos.clone();
    self.node_ids   = other.node_ids.clone();
    self.contexts   = other.contexts.clone();
  }

  pub fn reset(&mut self) -> Result<(), BoxedError> {
    log_trace!("reset");

    self.node_count   = 0;
    self.node_infos   = Vec::new();
    self.node_ids     = HashMap::new();
    self.contexts     = HashMap::new();

    Ok(())
  }

  pub fn node_id_from_name(&self, node_name : &str) -> Result<NodeId, BoxedError> {
    log_trace!("node_id_from_name");

    match self.node_ids.get(node_name) {
      Some(x) => Ok(*x),
      _       => {
        error!("node_id_from_name", "Node does not exist: `{}`", node_name)
      },
    }
  }

  pub fn node_info_from_id(&self, node_id : NodeId) -> Result<&NodeInfo, BoxedError> {
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
  pub fn graph_from(&mut self, context : &str) -> Result<&mut MeritRank, BoxedError> {
    log_trace!("graph_from: `{}`", context);

    if !self.contexts.contains_key(context) {
      if context.is_empty() {
        log_verbose!("Add context: NULL");
      } else {
        log_verbose!("Add context: `{}`", context);
      }

      let mut meritrank = MeritRank::new(Graph::new())?;

      for _ in 0..self.node_count {
        meritrank.get_new_nodeid();
      }

      self.contexts.insert(context.to_string(), meritrank);
    }

    match self.contexts.get_mut(context) {
      Some(graph) => Ok(graph),
      None        => {
        error!("graph_from", "Unable to add context `{}`", context)
      },
    }
  }

  pub fn find_or_add_node_by_name(
    &mut self,
    node_name : &str
  ) -> Result<NodeId, BoxedError> {
    log_trace!("find_or_add_node_by_name: `{}`", node_name);

    let node_id;

    if let Some(&id) = self.node_ids.get(node_name) {
      node_id = id;
    } else {
      node_id = self.node_count;

      self.node_count += 1;
      self.node_infos.resize(self.node_count, NodeInfo::default());
      self.node_infos[node_id] = NodeInfo {
        kind : kind_from_name(&node_name),
        name : node_name.to_string(),
      };
      self.node_ids.insert(node_name.to_string(), node_id);
    }

    for (context, meritrank) in &mut self.contexts {
      if meritrank.graph.contains_node(node_id) {
        continue;
      }

      if !context.is_empty() {
        log_verbose!("Add node in NULL: {}", node_id);
      } else {
        log_verbose!("Add node in `{}`: {}", context, node_id);
      }

      //  HACK!!!
      while meritrank.get_new_nodeid() < node_id {}
    }

    Ok(node_id)
  }

  pub fn set_edge(
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

      let null_weight = *self.graph_from("")?     .graph.edge_weight(src, dst).unwrap_or(None).unwrap_or(&0.0);
      let old_weight  = *self.graph_from(context)?.graph.edge_weight(src, dst).unwrap_or(None).unwrap_or(&0.0);
      let delta       = null_weight + amount - old_weight;

      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, delta);
      self.graph_from("")?.add_edge(src, dst, delta);

      log_verbose!("Add edge in `{}`: {} -> {} for {}", context, src, dst, amount);
      self.graph_from(context)?.add_edge(src, dst, amount);
    }

    Ok(())
  }

  pub fn connected_nodes(
    &mut self,
    context   : &str,
    ego       : NodeId
  ) -> Result<
    Vec<(NodeId, NodeId)>,
    BoxedError
  > {
    log_trace!("connected_nodes: `{}` {}", context, ego);

    let edge_ids : Vec<(NodeId, NodeId)> =
      all_neighbors(self.graph_from(context)?, ego)
        .into_iter()
        .map(|(dst_id, _)| (ego, dst_id))
        .collect();

    Ok(edge_ids)
  }

  pub fn connected_node_names(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Result<
    Vec<(&str, &str)>,
    BoxedError
  > {
    log_trace!("connected_node_names: `{}` `{}`", context, ego);

    let src_id   = self.node_id_from_name(ego)?;
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

  pub fn recalculate_all(&mut self, num_walk : usize) -> Result<(), BoxedError> {
    log_trace!("recalculate_all: {}", num_walk);

    let infos = self.node_infos.clone();
    let graph = self.graph_from("")?;

    for id in 0..infos.len() {
      if (id % 100) == 90 {
        log_trace!("{}%", (id * 100) / infos.len());
      }
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
        .filter(|(_, _, score)| score_gt < *score   || (score_gte && score_gt <= *score))
        .filter(|(_, _, score)| *score   < score_lt || (score_lte && score_lt >= *score))
        .collect::<Vec<(NodeId, NodeKind, Weight)>>()
        .into_iter()
        .filter(|(target_id, target_kind, _)| {
          if !hide_personal || (*target_kind != NodeKind::Comment && *target_kind != NodeKind::Beacon) {
            return true;
          }
          match self.graph_from(context) {
            Ok(graph) => match graph.graph.edge_weight(*target_id, node_id) {
              Ok(Some(_)) => false,
              _           => true,
            },
            Err(x) => {
              log_error!("(read_scores) {}", x);
              false
            },
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

    let src_id = self.find_or_add_node_by_name(src)?;
    let dst_id = self.find_or_add_node_by_name(dst)?;

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

    for (n, _) in all_neighbors(self.graph_from(context)?, id) {
      self.set_edge(context, id, n, 0.0)?;
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

    let focus_neighbors = all_neighbors(self.graph_from(context)?, focus_id);

    for (dst_id, focus_dst_weight) in focus_neighbors {
      let dst_kind = self.node_info_from_id(dst_id)?.kind;

      if dst_kind == NodeKind::User {
        if positive_only && edge_weight_or_zero(self.graph_from(context)?, ego_id, dst_id) <= 0.0 {
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
        let dst_neighbors = all_neighbors(self.graph_from(context)?, dst_id);

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
        match graph_cloned.get_node_data(node) {
          None       => Ok(None),
          Some(data) => {
            let kv : Vec<_> = data.neighbors(Neighbors::Positive).iter().skip(index).take(1).collect();

            if kv.is_empty() {
              Ok(None)
            } else {
              let n = kv[0].0;
              let w = kv[0].1;
             
              Ok(Some(Link::<NodeId, Weight> {
                neighbor       : *n,
                exact_distance : if w.abs() < 0.001 { 1_000_000.0 } else { 1.0 / w },
              }))
            }
          },
        }
      };

      let heuristic = |_node : NodeId| -> Result<Weight, BoxedError> {
        Ok(0.0)
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

        let a_b_weight = *self.graph_from(context)?.graph.edge_weight(a, b)?.ok_or("Got invalid edge")?;

        if k + 2 == ego_to_focus.len() {
          if a_kind == NodeKind::User {
            edges.push((a, b, a_b_weight));
          } else {
            log_trace!("ignore node {}", self.node_info_from_id(a)?.name);
          }
        } else if b_kind != NodeKind::User {
          log_trace!("ignore node {}", self.node_info_from_id(b)?.name);
          let c = ego_to_focus[k + 2];
          let b_c_weight = *self.graph_from(context)?.graph.edge_weight(b, c)?.ok_or("Got invalid edge")?;
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

      for (dst_id, weight) in all_neighbors(self.graph_from(context)?, src_id) {
        let dst_name =
          infos
            .get(dst_id)
            .ok_or("Node does not exist")?.name
            .as_str();
        v.push((src_name, dst_name, weight));
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

    let zero = self.find_or_add_node_by_name(ZERO_NODE.as_str())?;

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

    let _ = self.find_or_add_node_by_name(ZERO_NODE.as_str())?;

    self.recalculate_all(0)?; // FIXME Ad hok hack
    self.delete_from_zero()?;

    let nodes = self.top_nodes()?;

    self.recalculate_all(0)?; // FIXME Ad hok hack
    {
      let zero = self.node_id_from_name(ZERO_NODE.as_str())?;

      for (k, (node_id, amount)) in nodes.iter().enumerate() {
        if (k % 100) == 90 {
          log_trace!("{}%", (k * 100) / nodes.len());
        }
        self.set_edge("", zero, *node_id, *amount)?;
      }
    }
    self.recalculate_all(*NUM_WALK)?; // FIXME Ad hok hack

    return Ok(rmp_serde::to_vec(&())?);
  }
}
