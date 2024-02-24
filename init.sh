#!/bin/bash

set -m

# Start the primary process and put it in the background
/srv/meritrank-rust-service &

# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns
sleep 1

# Start the helper process
POSTGRES_DB_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres/postgres /srv/start

# now we bring the primary process back into the foreground
# and leave it there
# fg %1
fg
