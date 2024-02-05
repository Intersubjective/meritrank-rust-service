use std::env::var;
use lazy_static::lazy_static;
use nng::{Message, Protocol, Socket};
use reflection::Reflection;
use reflection_derive::Reflection;
use postgres::{Client, NoTls};

lazy_static! {
    static ref SERVICE_URL: String =
        var("RUST_SERVICE_URL")
            .unwrap_or("tcp://127.0.0.1:10234".to_string());

    static ref POSTGRES_DB_URL: String =
        var("POSTGRES_DB_URL")
            .unwrap_or("postgresql://postgres:postgres@localhost/postgres".to_string());
}

fn request<T: for<'a> serde::Deserialize<'a>>(
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

//#[derive(Deserialize,Reflection,Debug,PartialEq)]
#[derive(serde_derive::Deserialize,Reflection,Debug)]
struct Rec {
    subject: String,
    object: String,
    amount: i32 // f64
}

fn main() {
    println!("start (init) rust service graph...");

    let mut client = Client::connect(POSTGRES_DB_URL.as_str(), NoTls)
        .expect(format!("Cannot open DB: {}", *POSTGRES_DB_URL).as_str());

    for table in ["vote_user", "vote_beacon", "vote_comment"] {
        let sql = "SELECT subject, object, amount FROM ".to_owned() + table;
        let mut cnt = 0;
        for r in client.query(&sql, &[])
            .expect(format!("Cannot read table {table}").as_str())
        {
            let r = Rec {
                subject: r.get(0),
                object: r.get(1),
                amount: r.get(2),
            };
            mr_edge(r.subject.as_str(), r.object.as_str(), r.amount.into())
                .unwrap_or_else(|e| {
                    println!("Error adding {:?}, {:?}", r, e)
                });
            cnt = cnt + 1;
        }
        println!("{table}: {cnt}");
    }
}
