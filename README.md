# XHand Manipulation Workspace

ROS2 Humble workspace for Unitree G1 + XHand dexterous manipulation with RL.

## Prerequisites

- ROS2 Humble
- `legged_control2` Docker image (provides `legged_rl_controllers`, `legged_controllers`, `legged_bringup`, `unitree_description`, `mujoco_sim_ros2`)
- Python 3 with `mujoco` package (for URDF/MJCF generation)

## Structure

```
xhand_manip_ws/
├── src/
│   ├── g1_xhand_description/     # Combined G1+XHand model
│   │   ├── urdf/xhand_source/    # XHand left/right source URDFs + meshes
│   │   ├── scripts/              # combine_urdf.py, generate_mjcf.py
│   │   ├── config/               # controllers.yaml (53 joints)
│   │   └── launch/               # mujoco.launch.py
│   └── xhand_manip_controller/   # RL controller (extends OnnxController)
├── setup.sh                       # Run once after clone
└── README.md
```

## Quick Start

```bash
git clone git@github.com:jefferzn2001/xhand_manip_ws.git
cd xhand_manip_ws
bash setup.sh
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch g1_xhand_description mujoco.launch.py
```

## Regenerate Models

```bash
# Combined URDF (5mm spacer, 1.169kg measured mass per hand)
python3 src/g1_xhand_description/scripts/combine_urdf.py --measured-mass 1.169

# MuJoCo MJCF
python3 src/g1_xhand_description/scripts/generate_mjcf.py
```
