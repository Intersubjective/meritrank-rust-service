//  No context
pub const CMD_VERSION          : &str = "version";
pub const CMD_LOG_LEVEL        : &str = "log_level";
pub const CMD_SYNC             : &str = "sync";
pub const CMD_RESET            : &str = "reset";
pub const CMD_RECALCULATE_ZERO : &str = "recalculate_zero";
pub const CMD_NODE_LIST        : &str = "node_list";

//  With context
pub const CMD_NODE_SCORE       : &str = "node_score";
pub const CMD_SCORES           : &str = "scores";
pub const CMD_PUT_EDGE         : &str = "put_edge";
pub const CMD_DELETE_EDGE      : &str = "delete_edge";
pub const CMD_DELETE_NODE      : &str = "delete_node";
pub const CMD_GRAPH            : &str = "graph";
pub const CMD_CONNECTED        : &str = "connected";
pub const CMD_EDGES            : &str = "edges";
pub const CMD_MUTUAL_SCORES    : &str = "mutual_scores"; 

#[derive(Clone)]
pub struct Command {
  pub id       : String,
  pub context  : String,
  pub blocking : bool,
  pub payload  : Vec<u8>,
}

pub fn request_encode(_command : &Command) -> Result<Vec<u8>, ()> {
  todo!()
}

pub fn request_decode(_request : &[u8]) -> Result<Command, ()> {
  todo!()
}

pub fn response_encode<T>(_response : &T) -> Vec<u8> {
  todo!()
}

pub fn response_decode<T>(_response : &[u8]) -> T {
  todo!()
}
