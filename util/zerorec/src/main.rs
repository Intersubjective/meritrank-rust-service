use std::env::var;
use std::iter::Take;
use std::slice::Iter;
use lazy_static::lazy_static;
use nng::{Message, Protocol, Socket};
use serde::Deserialize;
use simple_pagerank::Pagerank;

lazy_static! {
    static ref SERVICE_URL: String =
        var("RUST_SERVICE_URL")
            .unwrap_or("tcp://127.0.0.1:10234".to_string());

    static ref ZERO_NODE: String =
        var("ZERO_NODE")
            .unwrap_or("Zero".to_string());

    static ref TOP_NODES_LIMIT: usize =
        var("TOP_NODES_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(100);
}

fn request<T: for<'a> Deserialize<'a>>(
    req: &Vec<u8>,
) -> Result<Vec<T>, Box<dyn std::error::Error + 'static>> {
    let client = Socket::new(Protocol::Req0)?;
    client.dial(&SERVICE_URL)?;
    client
        .send(Message::from(req.as_slice()))
        .map_err(|(_, err)| err)?;
    let msg: Message = client.recv()?;
    let slice: &[u8] = msg.as_slice();
    rmp_serde::from_slice(slice).or_else(|_| {
        let err: String = rmp_serde::from_slice(slice)?;
        Err(Box::from(format!("Server error: {}", err)))
    })
}

fn delete_edge(target: &str) -> Result<(), Box<dyn std::error::Error + 'static>>{
    let rq =
        ((("src", "delete", ZERO_NODE.as_str()), ("dest", "delete", target)), ());
    let req = rmp_serde::to_vec(&rq)?;
    let _res: Vec<()> = request(&req)?;
    Ok(())
}

fn delete_from_zero() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let rq_scores = ((("src", "=", ZERO_NODE.as_str()), ), ()); // mr_scores
    let req = rmp_serde::to_vec(&rq_scores)?;
    let res: Vec<(String, String, f64)> = request(&req)?;
    res
        .iter()
        .map(|(_, target, _)| {
            delete_edge(target.as_str())
        })
        .collect()
}

fn get_reduced_graph() -> Result<Vec<(String, String, f64)>, Box<dyn std::error::Error + 'static>>{
    let rq = ("for_beacons_global", ());
    let req = rmp_serde::to_vec(&rq)?;
    request(&req)
}

fn mr_edge(
    src: &str,
    dest: &str,
    weight: f64,
) -> Result<(), Box<dyn std::error::Error>,
> {
    let rq = (((src, dest, weight), ), ());
    let req = rmp_serde::to_vec(&rq)?;
    let _: Vec<(String, String, f64)> = request(&req)?;
    Ok(())
}

fn top_nodes() -> Vec<(String, f64)> {
    let reduced =
        get_reduced_graph()
            .expect("Cannot get reduced graph!");

    // PageRank
    let mut pr = Pagerank::<&String>::new();
    reduced
        .iter()
        .for_each(|(source, target, _weight)| {
            // TODO: check weight
            pr.add_edge(source, target);
        });
    pr.calculate();

    let (nodes, scores): (Vec<&&String>, Vec<f64>) =
        pr
            .nodes()    // already sorted by score
            .into_iter()
            .take(*TOP_NODES_LIMIT)
            .into_iter()
            .unzip();

    nodes
        .into_iter()
        .cloned()
        .cloned()
        .zip(scores)
        .collect::<Vec<_>>()
}

fn main() {
    println!("Zero node recalculation");
    println!("Using service at {}", *SERVICE_URL);

    // delete
    delete_from_zero().expect("Cannot delete old edges from Zero!");

    let nodes = top_nodes();
    nodes
        .iter()
        .for_each(|(node, score)| {
            println!("Creating edge from {} to {node} (score={score}) ...", *ZERO_NODE);
            let _ = mr_edge(ZERO_NODE.as_str(), node, *score)
                .map_err(|e| {
                    println!("Warning! Cannot create: {e}");
                    e
                });
        });

    println!("Zero re-calculator is over! Bye!")
}
