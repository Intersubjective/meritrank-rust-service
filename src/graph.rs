// Standard library imports
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use itertools::Itertools;

// External crate imports
use lazy_static::lazy_static;

// Current crate (`crate::`) imports
pub use crate::error::GraphManipulationError;
use crate::error::GraphManipulationError::UnknownContextFailure;
// use crate::logger::Logger;

// Current crate (`crate::`) imports
pub use meritrank::NodeId;
use meritrank::{MeritRank, MyGraph};

// Singleton instance
lazy_static! {
    pub static ref GRAPH: Arc<Mutex<GraphSingleton>> = Arc::new(Mutex::new(GraphSingleton::new()));
}

#[allow(dead_code)]
// GraphSingleton structure
pub struct GraphSingleton {
    graph: MyGraph, // null-context
    graphs: HashMap<String, MyGraph>, // contexted
    node_names: HashMap<String, NodeId>,
}

#[allow(dead_code)]
impl GraphSingleton {
    /// Constructor
    pub fn new() -> GraphSingleton {
        GraphSingleton {
            graph: MyGraph::new(),
            graphs: HashMap::new(),
            node_names: HashMap::new(),
        }
    }

    pub fn contexts() -> Result<Vec<String>, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(graph) => {
                let v = graph.graphs.keys().map(|ctx| ctx.clone()).collect_vec();
                Ok(v)
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    /// Get MeritRank object
    pub fn get_rank() -> Result<MeritRank, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(graph) => {
                let merit_rank = MeritRank::new(graph.graph.clone())?;
                Ok(merit_rank)
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    pub fn get_rank1(context: &str) -> Result<MeritRank, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(graph) => {
                let g =
                    graph.graphs.get(context).ok_or(UnknownContextFailure(context.to_string()))?;
                let merit_rank = MeritRank::new(g.clone())?;
                Ok(merit_rank)
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    /// Borrow Node Names
    pub fn borrow_node_names(&self) -> &HashMap<String, NodeId> {
        &self.node_names
    }

    /// Borrow Graph
    pub fn borrow_graph(&self) -> &MyGraph {
        &self.graph
    }

    pub fn borrow_graph0(&self, context: &String) -> Result<&MyGraph,GraphManipulationError> {
        if !self.graphs.contains_key(context) {
            //self.graphs.insert(context.clone(), MyGraph::new());
            Err(UnknownContextFailure(context.clone()))
        } else {
            Ok( self.graphs.get(context).expect("No context at borrow_graph1!") )
        }
    }
    pub fn borrow_graph1(&mut self, context: &String) -> &MyGraph {
        if !self.graphs.contains_key(context) {
            self.graphs.insert(context.clone(), MyGraph::new());
        }
        self.graphs.get(context).expect("No context at borrow_graph1!")
    }
    /// Borrow Graph Mut
    pub fn borrow_graph_mut(&mut self) -> &mut MyGraph {
        &mut self.graph
    }

    pub fn borrow_graph_mut1(&mut self, context: &String) -> &mut MyGraph {
        if !self.graphs.contains_key(context) {
            self.graphs.insert(context.clone(), MyGraph::new());
        }
        self.graphs.get_mut(context).expect("No context at  borrow_graph_mut1!")
    }

    // Node-related methods

    /// Creates a new node with the given name and returns its ID.
    /// If the node already exists, it returns the ID of the existing node.
    ///
    /// # Arguments
    ///
    /// * `node_name` - The name of the node to create or retrieve.
    ///
    /// # Errors
    ///
    /// Returns a `GraphManipulationError::MutexLockFailure()` if the mutex lock fails.
    pub fn add_node(node_name: &str) -> Result<NodeId, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(mut graph) => Ok(graph.get_node_id(node_name)),
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    // This method remains largely the same, it's already well structured
    pub fn get_node_id(&mut self, node_name: &str) -> NodeId {
        if let Some(&node_id) = self.node_names.get(node_name) {
            node_id
        } else {
            let new_node_id = self.graph.node_count() + 1;
            let node_id = NodeId::UInt(new_node_id);
            self.node_names.insert(node_name.to_string(), node_id);
            self.graph.add_node(node_id.into());
            node_id
        }
    }

    pub fn get_node_id1(&mut self, context: &str, node_name: &str) -> NodeId {
        if let Some(&node_id) = self.node_names.get(node_name) {
            // todo: what if the node exists in another context?
            node_id
        } else {
            let node_id = self.get_node_id(node_name); // create a node in null-context
            let graph = self.borrow_graph_mut1(&context.to_string());
            graph.add_node(node_id.into());
            node_id
        }
    }
    /// Returns the name of the node with the given ID.
    pub fn node_name_to_id_unsafe(&self, node_name: &str) -> Result<NodeId, GraphManipulationError> {
        if let Some(&node_id) = self.node_names.get(node_name) {
            Ok(node_id)
        } else {
            Err(GraphManipulationError::NodeNotFound(format!(
                "Node not found: {}",
                node_name
            )))
        }
    }
    pub fn node_name_to_id(node_name: &str) -> Result<NodeId, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(graph) => {
                graph.node_name_to_id_unsafe(node_name)
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    /// Returns the ID of the node with the given name.
    pub fn node_id_to_name_unsafe(&self, node_id: NodeId) -> Result<String, GraphManipulationError> {
        for (name, &id) in self.node_names.iter() {
            if id == node_id {
                return Ok(name.to_string());
            }
        }
        Err(GraphManipulationError::NodeNotFound(format!(
            "Node not found: {:?}",
            node_id
        )))
    }
    pub fn node_id_to_name(node_id: NodeId) -> Result<String, GraphManipulationError> {
        match GRAPH.lock() {
            Ok(graph) => {
                graph.node_id_to_name_unsafe(node_id)
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    pub fn clear_graph() -> Result<(), GraphManipulationError> {
        match GRAPH.lock() {
            Ok(mut graph) => {
                graph.graph.clear();
                graph.graphs.clear();
                graph.node_names.clear();
                Ok(())
            }
            Err(e) => Err(GraphManipulationError::MutexLockFailure(format!(
                "Mutex lock error: {}",
                e
            ))),
        }
    }

    // TODO: clear context?
}
