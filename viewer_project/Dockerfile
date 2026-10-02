FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxkbcommon-x11-0 \
    libxrender1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY viewer_project/requirements.txt /tmp/requirements.txt

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url ${TORCH_INDEX_URL} \
    && python3 -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

COPY viewer_project/ /app/
COPY vb_tactile_sensors/ /ros2_ws/src/vb_tactile_sensors/

WORKDIR /ros2_ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --base-paths src --install-base /ros2_ws/install --merge-install

COPY viewer_project/docker/entrypoint.sh /usr/local/bin/tactile-entrypoint
RUN chmod +x /usr/local/bin/tactile-entrypoint

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/tactile-entrypoint"]
CMD ["python3", "main.py"]