use std::env::var;
use lazy_static::lazy_static;
use nng::{Message, Protocol, Socket};

lazy_static! {
    static ref SERVICE_URL: String =
        var("RUST_SERVICE_URL")
            .unwrap_or("tcp://127.0.0.1:10234".to_string());
}

fn request(req: &Vec<u8>) -> Result<String, Box<dyn std::error::Error + 'static>> {
    let client = Socket::new(Protocol::Req0)?;
    client.dial(&SERVICE_URL)?;
    client
        .send(Message::from(req.as_slice()))
        .map_err(|(_, err)| err)?;
    let msg: Message = client.recv()?;
    let slice: &[u8] = msg.as_slice();
    return Ok(rmp_serde::from_slice(slice)?);
}

fn main() {
    println!("Zero node recalculation");
    println!("Using service at {}", *SERVICE_URL);

    let response = request(&rmp_serde::to_vec(&("zerorec", ())).unwrap()).unwrap();
    println!("Server: {}", response);

    println!("Done!")
}
