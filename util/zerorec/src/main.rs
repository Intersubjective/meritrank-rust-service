use std::env::var;
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
            .unwrap_or("U000000000000".to_string());

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

fn get_reduced_graph() -> Result<Vec<(String, String, f64)>, Box<dyn std::error::Error + 'static>> {
    let rq = ("for_beacons_global", ());
    let req = rmp_serde::to_vec(&rq)?;
    request(&req)
}

fn get_connected(ego: &str) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + 'static>> {
    let rq = (((ego, "connected"), ), ());
    let req = rmp_serde::to_vec(&rq)?;
    request(&req)
}

fn delete_edge(src: &str, target: &str) -> Result<(), Box<dyn std::error::Error + 'static>>{
    let rq =
        ((("src", "delete", src), ("dest", "delete", target)), ());
    let req = rmp_serde::to_vec(&rq)?;
    let _res: Vec<()> = request(&req)?;
    Ok(())
}

fn delete_from_zero() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let res: Vec<(String,String)> = get_connected(&ZERO_NODE)?;
    println!("Deleting {} old Zero edges...", res.len());
    let _ =
        res.
            iter()
            .map(| (src, target)| {
                assert!( ZERO_NODE.eq(src) || ZERO_NODE.eq(target) );
                delete_edge(src.as_str(), target.as_str())
            })
            .collect::<Result<Vec<_>, _>>()?;
    Ok(())
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
    let reduced: Vec<_> =
        get_reduced_graph()
            .expect("Cannot get reduced graph!");

    if reduced.is_empty() {
        eprintln!("Warning: reduced graph empty!");
        return Vec::new();
    }

    // PageRank
    let mut pr = Pagerank::<&String>::new();
    reduced
        .iter()
        .filter(|(source, target, _weight)|
            *source!=*ZERO_NODE && *target!=*ZERO_NODE
        )
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

    let nodes: Vec<(String, f64)> = top_nodes();
    if nodes.is_empty() {
        eprintln!("Top nodes are empty! Stop.");
        return;
    }
    println!("{} top nodes", nodes.len());

    // delete old Zero edges
    delete_from_zero().expect("Cannot delete old edges from Zero!");

    // create new Zero edges
    println!("Creating new Zero {} edges...", nodes.len());
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
