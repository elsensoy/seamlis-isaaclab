# Installation and Running

SEAMLiS runs in Docker on top of an Isaac Lab image. Isaac Sim, Isaac Lab, and
the Python dependencies are installed in the container; the host only needs the
GPU driver and container tooling.

Official references:

- [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)
- [Isaac Lab Docker guide](https://isaac-sim.github.io/IsaacLab/develop/source/features/docker_cloud.html)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Host requirements

- Ubuntu 22.04 or 24.04.
- An NVIDIA RTX GPU and a working production driver.
- Docker Engine 26.0.0 or newer.
- Docker Compose 2.25.0 or newer.
- NVIDIA Container Toolkit configured for Docker.
- At least 50 GB of free disk space.
- An NVIDIA NGC account if the Isaac Lab base image must be built from NGC.

Confirm that the host can see the GPU:

```bash
nvidia-smi
```

When Docker was installed through Snap, keep the Isaac Lab checkout below your
home directory, such as `~/IsaacLab`.

## 1. Install Docker

Install Docker using the official convenience script:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker "$USER"
newgrp docker
```

Verify Docker and Compose:

```bash
docker run --rm hello-world
docker compose version
```

## 2. Install the NVIDIA Container Toolkit

Add NVIDIA's package repository and install the toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access from a container:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Do not continue until this prints the GPU table.

## 3. Authenticate with NVIDIA NGC

The Isaac Lab base build may pull an Isaac Sim image from `nvcr.io`.

1. Sign in at [NVIDIA NGC](https://ngc.nvidia.com).
2. Generate a personal API key with NGC Catalog access.
3. Log in from the terminal:

```bash
docker login nvcr.io
```

Use `$oauthtoken` literally as the username and the generated `nvapi-...` key
as the password.

## 4. Clone and build Isaac Lab

Clone Isaac Lab under the home directory:

```bash
cd "$HOME"
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

Check the container helper supported by the checked-out Isaac Lab version:

```bash
python3 docker/container.py --help
```

Build and start its base container. The repository version currently used by
SEAMLiS supports:

```bash
python3 docker/container.py start base
```

If the checked-out version documents `./docker/container.py start` instead,
use that command. Confirm the resulting image name:

```bash
docker images | grep -i isaac
```

The SEAMLiS `Dockerfile` currently inherits from `isaac-lab-base-elida`. If the
helper produced `isaac-lab-base`, add the expected local tag:

```bash
docker tag isaac-lab-base isaac-lab-base-elida
```

## 5. Configure the project checkout

Place SEAMLiS next to Isaac Lab when possible:

```text
~/IsaacLab/
~/seamlis/
```

In the SEAMLiS repository, create or update `.env` so Compose can mount the
Isaac Lab source directory:

```dotenv
ISAACLAB_SOURCE=../IsaacLab/source
```

An absolute path also works:

```dotenv
ISAACLAB_SOURCE=/home/your-user/IsaacLab/source
```

The source is mounted read-only at `/workspace/isaaclab/source`, while the
SEAMLiS checkout is mounted read-write at `/workspace/seamlis`. Host edits are
therefore visible immediately inside the development container.

## 6. Build the SEAMLiS image

From the SEAMLiS repository:

```bash
docker compose build seamlis-dev
```

Show full build output when diagnosing a failure:

```bash
BUILDKIT_PROGRESS=plain docker compose build seamlis-dev
```

## 7. Start the development container

For headless work:

```bash
docker compose up -d seamlis-dev
docker compose exec seamlis-dev bash
```

For the Isaac viewport or Matplotlib windows, first allow the local Docker user
to connect to X11, then start the service:

```bash
xhost +local:docker
docker compose up -d seamlis-dev
docker compose exec seamlis-dev bash
```

The Compose service passes the display, Xauthority file, X11 socket, and GPU
through to the container. It runs as the image's `ubuntu` user so its cache and
Xauthority paths remain consistent.

## 8. Run an experiment

Use Isaac Sim's Python wrapper for project scripts:

```bash
ISAAC_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
```

### Headless Isaac run

```bash
"$ISAAC_PYTHON" scripts/run_isaac.py \
  --config configs/test_1_blind_corner_visibility_area.yaml \
  --headless
```

### Isaac viewport only

```bash
"$ISAAC_PYTHON" scripts/run_isaac.py \
  --config configs/test_1_blind_corner.yaml \
  --viz kit
```

### Isaac viewport and exploration UI

```bash
"$ISAAC_PYTHON" scripts/run_isaac.py \
  --config configs/test_1_blind_corner_visibility_area.yaml \
  --explore-ui \
  --viz kit
```

### RTX LiDAR visualization

```bash
"$ISAAC_PYTHON" scripts/run_isaac.py \
  --config configs/test_1_blind_corner.yaml \
  --explore-ui \
  --use-rtx-lidar \
  --viz kit
```

`--use-rtx-lidar` cannot be combined with `--low-memory`.

### Viewport smoke test

Use the small demo before a longer visual run:

```bash
"$ISAAC_PYTHON" tests/demo.py --viz kit
```

## 9. Run the headless safety benchmark

The benchmark compares the configured scenarios, exploration algorithms, and
attitude policies without creating an Isaac viewport or Matplotlib figure:

```bash
"$ISAAC_PYTHON" examples/benchmark_config_suite.py
```

Run a short comparison first:

```bash
"$ISAAC_PYTHON" examples/benchmark_config_suite.py \
  --scenario blind_corner \
  --algorithm Frontier \
  --attitude visibility_area \
  --attitude gatekeeper \
  --max-steps 20 \
  --output-dir output/headless_safety_smoke
```

See [`configs/benchmarks/headless_safety/README.md`](../configs/benchmarks/headless_safety/README.md)
for suite filters, repeats, metrics, and output files.

## Command-line arguments

| Argument | Description |
| --- | --- |
| `--config`, `-c` | Required YAML experiment configuration. |
| `--headless` | Run Isaac Sim without a viewport. |
| `--viz kit` | Open the Isaac Sim viewport. |
| `--explore-ui` | Open the 2D Matplotlib planner view. |
| `--use-rtx-lidar` | Enable RTX LiDAR; incompatible with `--low-memory`. |
| `--low-memory` | Reduce renderer resolution and memory consumption. |

Run the script with `--help` to see Isaac Lab launcher arguments supplied by
the installed version:

```bash
"$ISAAC_PYTHON" scripts/run_isaac.py --help
```

## Stop the environment

Exit the container shell and stop the Compose services:

```bash
exit
docker compose down
```

Named cache volumes remain available for later runs.

## Troubleshooting

### Base image not found

If the build reports that `isaac-lab-base` does not exist, build the
Isaac Lab base image and apply the local tag from step 4.

### GPU is unavailable in the container

Run:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If it fails, reconfigure and restart the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build exits with `apt-get` permission errors

The base image uses the non-root `ubuntu` user. System packages and Python
packages installed into Isaac's bundled environment must be installed before
the Dockerfile's final `USER ubuntu` line.

### EULA warning

Compose sets `ACCEPT_EULA=Y`. For a one-off `docker run`, pass it explicitly:

```bash
docker run --rm -e ACCEPT_EULA=Y IMAGE COMMAND
```

### Display or cache permission errors

Do not add a host `user: UID:GID` override unless all cache and Xauthority
mounts are updated for that user. The current Compose file and image are
configured for the container's `ubuntu` user.

### Black viewport or missing windows

Check `DISPLAY`, allow the X11 connection, and restart the service:

```bash
echo "$DISPLAY"
xhost +local:docker
docker compose restart seamlis-dev
```

### Hidden base-image build failure

Re-run the Isaac Lab build with plain progress and inspect disk usage:

```bash
BUILDKIT_PROGRESS=plain python3 docker/container.py start base
df -h /
docker system df
```


