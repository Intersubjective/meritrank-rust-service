# MeritRank service (NNG server)
NNG server for https://github.com/Intersubjective/meritrank-psql-connector with embedded Rust MeritRank engine.

See also:
https://github.com/shestero/pgmer2

NB The lib_graph code was originally taken from
https://github.com/vsradkevich/pg_meritrank/tree/main/src/lib_graph

## Env variables
- `MERITRANK_SERVICE_URL` - default `"tcp://127.0.0.1:10234"`
- `MERITRANK_SERVICE_THREADS` - default `1`
- `MERITRANK_NUM_WALK` - default `10000`
- `MERITRANK_WEIGHT_MIN_LEVEL` - default `0.1`
- `MERITRANK_ZERO_NODE` - default `U000000000000`
- `MERITRANK_TOP_NODES_LIMIT` - default `100`

## Dev setup for manual testing
- Make sure to run tests sequentially.
```sh
export RUST_TEST_THREADS=1
cargo test
```
