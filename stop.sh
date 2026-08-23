#!/usr/bin/env bash
set -e

# Stop the docker-compose service
docker compose down

# Revoke X11 access
xhost -local:docker >/dev/null
 