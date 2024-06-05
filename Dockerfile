FROM rust:1.75.0-alpine as compile
RUN apk add cmake make gcc musl-dev
WORKDIR /usr/project
COPY . .
RUN cargo build --release

FROM scratch as release
EXPOSE 10234
ENV RUST_SERVICE_PARALLEL=128
WORKDIR /srv
COPY --from=compile /usr/project/target/release/meritrank-service meritrank-service
ENTRYPOINT [ "/srv/meritrank-service" ]
