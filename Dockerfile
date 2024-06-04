FROM rust:1.75.0-alpine as compile
#FROM rust:1.74.1-bookworm as compile
RUN apk add cmake make gcc musl-dev
#RUN apt-get update && apt-get -y install cmake
# RUN cargo build # --target x86_64-unknown-linux-gnu # debug
WORKDIR /usr/project
COPY . .
RUN cargo build && cargo build --release

FROM alpine as debug
#FROM scratch as debug
RUN apk add bash
WORKDIR /srv
COPY --from=compile /usr/project/target/debug/meritrank-rust-service meritrank-rust-service
#COPY --from=compile /lib/x86_64-linux-gnu/libgcc_s.so.1 .
ENV RUST_SERVICE_URL=tcp://0.0.0.0:10234
EXPOSE 10234
ENV LD_LIBRARY_PATH=.

FROM alpine as release
#FROM scratch as release
RUN apk add bash
WORKDIR /srv
COPY --from=compile /usr/project/target/release/meritrank-rust-service meritrank-rust-service
#COPY --from=compile /lib/x86_64-linux-gnu/libgcc_s.so.1 .
ENV RUST_SERVICE_PARALLEL=128
ENV RUST_SERVICE_URL=tcp://0.0.0.0:10234
EXPOSE 10234
ENV LD_LIBRARY_PATH=.
