FROM rust:1.74.1-alpine as compile
#FROM rust:1.74.1-bookworm as compile
WORKDIR /usr/project
COPY . .
RUN apk add cmake make gcc musl-dev
#RUN apt-get update && apt-get -y install cmake
RUN cargo build # --target x86_64-unknown-linux-gnu # debug
RUN cargo build --release # --target x86_64-unknown-linux-gnu
RUN cd util/zerorec && cargo build && cargo build --release && cd ..
RUN cd util/start && cargo build && cargo build --release && cd ..

FROM alpine as debug
#FROM scratch as debug
WORKDIR /srv
COPY --from=compile /usr/project/target/debug/meritrank-rust-service meritrank-rust-service
COPY --from=compile /usr/project/util/zerorec/target/debug/zerorec zerorec
#COPY --from=compile /lib/x86_64-linux-gnu/libgcc_s.so.1 .
# ENV RUST_SERVICE_PARALLEL=128
ENV RUST_SERVICE_URL=tcp://0.0.0.0:10234
EXPOSE 10234
ENV LD_LIBRARY_PATH=.
ENTRYPOINT ["/srv/meritrank-rust-service"]

FROM alpine as release
#FROM scratch as release
WORKDIR /srv
COPY --from=compile /usr/project/target/release/meritrank-rust-service meritrank-rust-service
COPY --from=compile /usr/project/util/zerorec/target/release/zerorec zerorec
#COPY --from=compile /lib/x86_64-linux-gnu/libgcc_s.so.1 .
ENV RUST_SERVICE_PARALLEL=128
ENV RUST_SERVICE_URL=tcp://0.0.0.0:10234
EXPOSE 10234
ENV LD_LIBRARY_PATH=.
ENTRYPOINT ["/srv/meritrank-rust-service"]
