# Task Tracker

## Completed

- [x] **Fix robot falling on spawn** (2026-03-06)
  - Root cause: 3-way mismatch between MJCF initial state, ros2_control.xacro, and controllers.yaml
  - Added MuJoCo keyframe, synced initial positions, adjusted pelvis height
  - Note: keyframe not auto-applied by mujoco_sim_ros2 (uses mj_resetData, not mj_resetDataKeyframe)

- [x] **Write standby policy plan** (2026-03-06)
  - Created `STANDBY_POLICY_PLAN.md` with full spec: 15 joints, 65-dim obs, reward function, domain randomization, deployment path

## In Progress

- [ ] **Train RL standby policy** — Replace fixed PD standby controller with trained standing policy (see `STANDBY_POLICY_PLAN.md`)

## Backlog

- [ ] Build `StandbyPolicyController` C++ class (extend OnnxController) for deploying the trained ONNX policy
- [ ] Fix mujoco_sim_ros2 init — build from source with `mj_resetDataKeyframe` call, or write init plugin

## Discovered During Work

- The installed `mujoco_ros2_control` plugin does NOT support the `initial_keyframe` parameter (that feature exists in newer upstream versions)
- `mujoco_sim_ros2` calls `mj_makeData` + `mj_forward` but never `mj_resetDataKeyframe`, so MJCF keyframes are ignored at startup
