//  ================================================================
//
//    Iterative general implementation for
//    the A* graph search algorithm
//
//  ================================================================
//
//    (C) 2024 Mitya Selivanov <guattari.tech>, The MIT License
//
//  ================================================================

#[allow(non_camel_case_types)]
mod astar {
  use std::{ops::Add, fmt::Debug};
  
  #[derive(Debug, Clone, PartialEq)]
  pub enum Status {
    PROGRESS,
    SUCCESS,
    FAIL,
  }

  #[derive(Debug, Clone, Default)]
  pub struct Link<Node_Id, Cost> {
    pub neighbor       : Node_Id,
    pub exact_distance : Cost,
  }

  #[derive(Debug, Clone, Default)]
  pub struct Node<Node_Id, Cost> {
    pub id             : Node_Id,
    pub previous       : Option<Node_Id>,
    pub exact_distance : Cost,
    pub estimate       : Cost,
    pub count          : usize,
  }

  pub struct State<Node_Id, Cost> {
    pub open             : Vec<Node<Node_Id, Cost>>,
    pub closed           : Vec<Node<Node_Id, Cost>>,
    pub source           : Node_Id,
    pub destination      : Node_Id,
    pub closest_index    : usize,
    pub closest_estimate : Cost,
  }

  pub fn init<Node_Id, Cost>(
    source      : Node_Id,
    destination : Node_Id,
    max_cost    : Cost
  ) -> State<Node_Id, Cost>
    where
      Node_Id : Clone,
      Cost    : Clone + Default
  {
    let source_node = Node::<Node_Id, Cost> {
      id             : source.clone(),
      previous       : None,
      exact_distance : Cost::default(),
      estimate       : max_cost.clone(),
      count          : 1,
    };

    let mut open   : Vec<Node<Node_Id, Cost>> = vec![source_node];
    let mut closed : Vec<Node<Node_Id, Cost>> = vec![];

    open.reserve(128);
    closed.reserve(128);

    return State {
      open,
      closed,
      source,
      destination,
      closest_index    : 0,
      closest_estimate : max_cost,
    };
  }

  pub fn path<Node_Id, Cost>(
    state : &mut State<Node_Id, Cost>
  ) -> Option<Vec<Node_Id>>
    where
      Node_Id : Clone + PartialEq
  {
    if state.closed.is_empty() || state.closest_index >= state.closed.len() {
      return None;
    }

    let mut current = state.closest_index;

    let mut backward : Vec<Node_Id> = vec![state.closed[current].id.clone()];
    backward.reserve_exact(state.closed[current].count);

    loop {
      if backward[backward.len() - 1] == state.source {
        break;
      }

      if backward.len() > state.closed.len() {
        return None;
      }

      let mut index = usize::MAX;
      for i in 0..state.closed.len() {
        if Some(state.closed[i].id.clone()) == state.closed[current].previous {
          index = i;
          break;
        }
      }

      if index == usize::MAX {
        return None;
      }

      backward.push(state.closed[index].id.clone());
      current = index;
    }

    let mut forward = Vec::<Node_Id>::new();
    forward.reserve_exact(backward.len());

    for i in 1..=backward.len() {
      forward.push(backward[backward.len() - i].clone());
    }

    return Some(forward);
  }

  pub fn iteration<Node_Id, Cost, Neighbor, Heuristic>(
    state     : &mut State<Node_Id, Cost>,
    neighbor  : Neighbor,
    heuristic : Heuristic
  ) -> Status
    where
      Node_Id   : Debug + Clone + Default + PartialEq,
      Cost      : Debug + Clone + Default + PartialOrd + Add<Output = Cost>,
      Neighbor  : Fn(Node_Id, usize) -> Option<Link<Node_Id, Cost>>,
      Heuristic : Fn(Node_Id)        -> Cost
  {
    if state.open.is_empty() {
      return Status::FAIL;
    }

    //  Find the nearest node to the destination in the open set
    //

    let mut index_in_open : usize = 0;
    for index in 1..state.open.len() {
      if state.open[index].estimate < state.open[index_in_open].estimate {
        index_in_open = index;
      }
    }

    let nearest_node = state.open[index_in_open].clone();
    if index_in_open != state.open.len() - 1 {
      state.open[index_in_open] = state.open[state.open.len() - 1].clone();
    }
    state.open.resize(state.open.len() - 1, Node::default());

    //  Check if we reached the destination
    //
    if nearest_node.id == state.destination {
      state.closed.push(nearest_node);
      state.closest_index    = state.closed.len() - 1;
      state.closest_estimate = Cost::default();

      //  Finish the search
      return Status::SUCCESS;
    }

    //  Enumerate all neighbors
    //
    let mut neighbor_index : usize = 0;
    loop {
      //  Get a link to the neighbor node
      //
      let link = match neighbor(nearest_node.clone().id, neighbor_index) {
        Some(x) => x,
        None    => break, // There is no more neighbors, so end the loop.
      };

      //  Calculate distance estimations
      //

      let exact_distance = nearest_node.clone().exact_distance + link.clone().exact_distance;
      let estimate       = heuristic(link.clone().neighbor);

      let neighbor_node = Node {
        id             : link.neighbor,
        previous       : Some(nearest_node.clone().id),
        exact_distance,
        estimate,
        count          : nearest_node.count + 1,
      };

      //  Check if we reached the destination
      //
      if neighbor_node.id == state.destination {
        state.closed.push(nearest_node);
        state.closed.push(neighbor_node.clone());
        state.closest_index    = state.closed.len() - 1;
        state.closest_estimate = Cost::default();

        //  Finish the search
        return Status::SUCCESS;
      }

      //  Check if this node is already in the closed set
      //

      let mut index_in_closed = usize::MAX;
      for i in 0..state.closed.len() {
        if state.closed[i].id == neighbor_node.id {
          index_in_closed = i;
          break;
        }
      }

      if index_in_closed != usize::MAX {
        //  Check if this node has a better distance
        if neighbor_node.exact_distance < state.closed[index_in_closed].exact_distance {
          if neighbor_node.estimate < state.closest_estimate {
            state.closest_index    = index_in_closed;
            state.closest_estimate = neighbor_node.clone().estimate;
          }

          //  Replace the node
          state.closed[index_in_closed] = neighbor_node;
        }

        //  Skip this node
        neighbor_index += 1;
        continue;
      }

      //  Check if this node is already in the open set
      //

      let mut index_in_open : usize = usize::MAX;
      for i in 0..state.open.len() {
        if state.open[i].id == neighbor_node.id {
          index_in_open = i;
          break;
        }
      }

      if index_in_open != usize::MAX {
        //  Check if this node has a better distance
        if neighbor_node.exact_distance < state.open[index_in_open].exact_distance {
          //  Replace the node
          state.open[index_in_open] = neighbor_node;
        }

        //  Skip this node
        neighbor_index += 1;
        continue;
      }

      state.open.push(neighbor_node);

      //  Proceed to the next neighbor node
      neighbor_index += 1;
    }

    if nearest_node.estimate < state.closest_estimate {
      state.closest_index    = state.closed.len();
      state.closest_estimate = nearest_node.clone().estimate;
    }

    state.closed.push(nearest_node);

    return Status::PROGRESS;
  }
}

//  ================================================================
//
//    Testing
//
//  ================================================================

#[cfg(test)]
mod tests {
  use super::astar::*;

  #[test]
  fn path_exists() {
    let graph : Vec<((i64, i64), i64)> = vec![
      ((0, 1),  5),
      ((0, 2),  3),
      ((1, 3),  4),
      ((2, 4),  1),
      ((3, 5), 10),
      ((4, 6),  1),
      ((6, 7),  1),
      ((7, 5),  1),
    ];

    let neighbor = |id : i64, index : usize| {
      let mut k : usize = 0;
      for ((src, dst), cost) in graph.clone() {
        if src == id {
          if k == index {
            return Some(Link::<i64, i64> {
              neighbor       : dst,
              exact_distance : cost
            });
          } else {
            k += 1;
          }
        }
      }
      return None;
    };

    let heuristic = |id : i64| -> i64 {
      return (8 - id).abs();
    };

    let mut state = init(0i64, 5i64, i64::MAX);

    let mut steps = 0;
    loop {
      steps += 1;
      let status = iteration(&mut state, neighbor, heuristic);
      if status != Status::PROGRESS {
        assert_eq!(status, Status::SUCCESS);
        break;
      }
    }

    let v = path(&mut state).unwrap();

    assert_eq!(steps, 5);
    assert_eq!(v.len(), 6);
    assert_eq!(v[0], 0);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 4);
    assert_eq!(v[3], 6);
    assert_eq!(v[4], 7);
    assert_eq!(v[5], 5);
  }

  #[test]
  fn path_does_not_exist() {
    let graph : Vec<((i64, i64), i64)> = vec![
      ((0, 1),  5),
      ((0, 2),  3),
      ((1, 3),  4),
      ((2, 4),  1),
      ((3, 5),  1),
      ((4, 6), 10),
      ((6, 7),  1),
      ((7, 5),  1),
    ];

    let neighbor = |id : i64, index : usize| {
      let mut k : usize = 0;
      for ((src, dst), cost) in graph.clone() {
        if src == id {
          if k == index {
            return Some(Link::<i64, i64> {
              neighbor       : dst,
              exact_distance : cost
            });
          } else {
            k += 1;
          }
        }
      }
      return None;
    };

    let heuristic = |id : i64| -> i64 {
      return (15 - id).abs();
    };

    let mut state = init(0i64, 15i64, i64::MAX);

    let mut steps = 0;
    loop {
      steps += 1;
      let status = iteration(&mut state, neighbor, heuristic);
      if status != Status::PROGRESS {
        assert_eq!(status, Status::FAIL);
        break;
      }
    }

    let v = path(&mut state).unwrap();

    assert_eq!(steps, 9);
    assert_eq!(v.len(), 5);
    assert_eq!(v[0], 0);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 4);
    assert_eq!(v[3], 6);
    assert_eq!(v[4], 7);
  }

  #[test]
  fn empty_path() {
    let graph : Vec<((i64, i64), i64)> = vec![
      ((0, 1),  5),
      ((0, 2),  3),
      ((1, 3),  4),
      ((2, 4),  1),
      ((3, 5),  1),
      ((4, 6), 10),
      ((6, 7),  1),
      ((7, 5),  1),
    ];

    let neighbor = |id : i64, index : usize| {
      let mut k : usize = 0;
      for ((src, dst), cost) in graph.clone() {
        if src == id {
          if k == index {
            return Some(Link::<i64, i64> {
              neighbor       : dst,
              exact_distance : cost
            });
          } else {
            k += 1;
          }
        }
      }
      return None;
    };

    let heuristic = |id : i64| -> i64 {
      return (2 - id).abs();
    };

    let mut state = init(2i64, 2i64, i64::MAX);

    let mut steps = 0;
    loop {
      steps += 1;
      let status = iteration(&mut state, neighbor, heuristic);
      if status != Status::PROGRESS {
        assert_eq!(status, Status::SUCCESS);
        break;
      }
    }

    let v = path(&mut state).unwrap();

    assert_eq!(steps, 1);
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 2);
  }

  #[test]
  fn cyclic() {
    let graph : Vec<((i64, i64), i64)> = vec![
      ((0, 1),  5),
      ((0, 2),  3),
      ((1, 3),  4),
      ((2, 4),  1),
      ((3, 5),  1),
      ((4, 6), 10),
      ((6, 7),  1),
      ((7, 5),  1),
      ((7, 0),  5),
      ((5, 1),  5),
      ((6, 2),  5),
    ];

    let neighbor = |id : i64, index : usize| {
      let mut k : usize = 0;
      for ((src, dst), cost) in graph.clone() {
        if src == id {
          if k == index {
            return Some(Link::<i64, i64> {
              neighbor       : dst,
              exact_distance : cost
            });
          } else {
            k += 1;
          }
        }
      }
      return None;
    };

    let heuristic = |id : i64| -> i64 {
      return (5 - id).abs();
    };

    let mut state = init(0i64, 5i64, i64::MAX);

    let mut steps = 0;
    loop {
      steps += 1;
      let status = iteration(&mut state, neighbor, heuristic);
      if status != Status::PROGRESS {
        assert_eq!(status, Status::SUCCESS);
        break;
      }
    }

    let v = path(&mut state).unwrap();

    assert_eq!(steps, 5);
    assert_eq!(v.len(), 6);
    assert_eq!(v[0], 0);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 4);
    assert_eq!(v[3], 6);
    assert_eq!(v[4], 7);
    assert_eq!(v[5], 5);
  }
}
