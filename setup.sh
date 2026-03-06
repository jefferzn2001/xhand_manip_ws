#!/bin/bash
# Setup script: creates symlinks and regenerates models.
# Run once after cloning the repo.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESC_DIR="$SCRIPT_DIR/src/g1_xhand_description"

echo "=== XHand Manipulation Workspace Setup ==="

# 1. Symlink G1 meshes from system unitree_description
G1_MESHES="/opt/ros/humble/share/unitree_description/meshes/g1"
if [ -d "$G1_MESHES" ]; then
    ln -sfn "$G1_MESHES" "$DESC_DIR/meshes/g1"
    echo "[OK] Symlinked G1 meshes from $G1_MESHES"
else
    echo "[WARN] G1 meshes not found at $G1_MESHES"
    echo "       Install unitree_description or set up manually"
fi

# 2. Regenerate combined URDF (requires G1 URDF from system)
if [ -f "/opt/ros/humble/share/unitree_description/urdf/g1/main.urdf" ]; then
    echo "Generating combined G1+XHand URDF..."
    python3 "$DESC_DIR/scripts/combine_urdf.py" --measured-mass 1.169
else
    echo "[WARN] G1 URDF not found, skipping URDF generation"
fi

# 3. Copy XHand meshes into description meshes/ (from xhand source URDFs' sibling dirs)
# The STL files are committed in the repo under meshes/
echo "[OK] XHand meshes already in src/g1_xhand_description/meshes/"

# 4. Regenerate MJCF (modifies stock G1 MJCF, adds XHand bodies)
echo "Generating MuJoCo MJCF..."
python3 "$DESC_DIR/scripts/generate_mjcf.py"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  source /opt/ros/humble/setup.bash"
echo "  colcon build --symlink-install"
echo "  source install/setup.bash"
echo "  ros2 launch g1_xhand_description mujoco.launch.py"
