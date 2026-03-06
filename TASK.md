# Task Tracker

## Completed

- [x] **Fix robot falling on spawn** (2026-03-06)
  - Root cause: 3-way mismatch between MJCF initial state, ros2_control.xacro, and controllers.yaml
  - Added MuJoCo keyframe, synced initial positions, adjusted pelvis height
  - Note: keyframe not auto-applied by mujoco_sim_ros2 (uses mj_resetData, not mj_resetDataKeyframe)

- [x] **Write standby policy plan** (2026-03-06)
  - Created `STANDBY_POLICY_PLAN.md` with full spec: 15 joints, 65-dim obs, reward function, domain randomization, deployment path

- [x] **Write Isaac Lab adaptation analysis** (2026-03-06)
  - Added `ISAACLAB.md` as a single handoff document analyzing `whole_body_tracking`
  - Summarized what can be reused, what must change for a true standby task, and how to deploy the trained ONNX in this ROS 2 workspace
  - Added `/whole_body_tracking/` to the parent `.gitignore` so the nested repo is not tracked by this workspace

## In Progress

- [ ] **Train RL standby policy** — Replace fixed PD standby controller with trained standing policy (see `STANDBY_POLICY_PLAN.md`)

## Backlog

- [ ] Build `StandbyPolicyController` C++ class (extend OnnxController) for deploying the trained ONNX policy
- [ ] Fix mujoco_sim_ros2 init — build from source with `mj_resetDataKeyframe` call, or write init plugin
- [ ] Create Isaac Lab `tasks/standby/` by forking the `whole_body_tracking` tracking task and reducing it to a 15-joint standing task

## Discovered During Work

- The installed `mujoco_ros2_control` plugin does NOT support the `initial_keyframe` parameter (that feature exists in newer upstream versions)
- `mujoco_sim_ros2` calls `mj_makeData` + `mj_forward` but never `mj_resetDataKeyframe`, so MJCF keyframes are ignored at startup
- `whole_body_tracking` already matches several standby requirements well: 50 Hz policy cadence, G1 actuator config, randomization hooks, PPO runner, and ONNX export
