FROM rust:1.75.0-alpine as compile
RUN apk add cmake make gcc musl-dev
WORKDIR /usr/project
COPY . .
RUN cargo build --release

FROM scratch
EXPOSE 10234
ENV MERITRANK_SERVICE_URL=tcp://127.0.0.1:10234
ENV MERITRANK_SERVICE_THREADS=32
ENV MERITRANK_NUM_WALK=10000
ENV MERITRANK_ZERO_NODE=U000000000000
ENV MERITRANK_TOP_NODES_LIMIT=100
WORKDIR /srv
ENTRYPOINT [ "/srv/meritrank-service" ]
COPY --from=compile /usr/project/target/release/meritrank-service meritrank-service
