#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export QT_QPA_PLATFORM_PLUGIN_PATH="$(
    python3 -c 'from PyQt6.QtCore import QLibraryInfo; print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))'
)"
exec "$@"
