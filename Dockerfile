# Base image with Isaac Lab already installed
FROM isaac-lab-base-elida

# Noninteractive + kit root
ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1

#  System deps (combined into ONE layer) 
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig fontconfig-config libfontconfig1 libfreetype6 fonts-dejavu-core \
    libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libxi6 \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
    libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xinput0 libxcb-randr0 \
    libxkbcommon0 libxkbcommon-x11-0 libgl1 libglib2.0-0 libdbus-1-3 \
    ffmpeg \
 && fc-cache -f \
 && rm -rf /var/lib/apt/lists/*

# Workdir where seamlis repo will be MOUNTED at runtime
WORKDIR /workspace/seamlis

# Make Isaac Lab + seamlis importable
ENV PYTHONPATH=/workspace/isaaclab/source:/workspace/isaaclab/source/isaaclab:/workspace/seamlis

# Qt/X11 settings
ENV QT_PLUGIN_PATH= \
    QT_QPA_PLATFORM=xcb \
    XDG_RUNTIME_DIR=/tmp/xdg-runtime \
    QT_QPA_PLATFORM_PLUGIN_PATH=/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.12/site-packages/PyQt5/Qt5/plugins/platforms

# ---- Python deps (installed into Isaac's Python) ----
RUN /bin/bash -lc '\
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir --no-deps shapely trimesh && \
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir --no-deps scikit-fmm casadi "do-mpc" && \
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir pandas && \
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir PyQt5==5.15.11 PyQt5-sip==12.17.0 \
'

# Various libraries (matplotlib, warp, fontconfig, ...) create their own
# cache dirs under $HOME/.cache on first use. Make sure ubuntu can write
# there -- the ov/pip/kit subdirs get overlaid by named volumes at runtime
# (chowned separately), but the parent itself is a plain image directory
# and needs to be owned by ubuntu for those first-use mkdirs to succeed.
RUN mkdir -p /home/ubuntu/.cache /tmp/xdg-runtime \
 && chown -R ubuntu:ubuntu /home/ubuntu /tmp/xdg-runtime \
 && chmod 700 /tmp/xdg-runtime

# Drop to the unprivileged user only now that everything above (apt, pip
# installs into Isaac's bundled Python) is done as root -- those installs
# write into root-owned site-packages and silently no-op for other users
# if this comes any earlier.
USER ubuntu

# Default entrypoint and command
ENTRYPOINT ["/workspace/isaaclab/_isaac_sim/python.sh"]
CMD ["/workspace/seamlis/temp.py"]
