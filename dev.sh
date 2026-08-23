# dev.sh  
#!/usr/bin/env bash
set -e
xhost +local:docker >/dev/null
docker compose up -d --build
docker compose exec seamlis-dev bash
