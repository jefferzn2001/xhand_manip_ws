# Isaac Lab Standby Controller Plan

## Goal
Use the existing `whole_body_tracking` Isaac Lab project as the base for a new G1 + XHand standing task, then deploy the trained policy back into this ROS 2 workspace as a true standby controller.

This document is intentionally written as a single handoff file for future search / GPT-assisted follow-up.

## Executive Summary
- The existing `whole_body_tracking` codebase is a strong starting point for a standby controller.
- The adaptation difficulty is `easy-to-medium`, not hard.
- The best path is not to build a brand-new Isaac Lab stack from scratch.
- The best path is to fork the existing tracking task into a new standing task and simplify it.
- The existing project already gives us the hard parts: G1 robot config, actuator tuning, Isaac Lab task registration, PPO training script, ONNX export, and domain randomization patterns.
- The main work is replacing motion-tracking commands/rewards/terminations with standing-specific ones and reducing the action space from whole-body to lower-body plus waist.

## What Exists In `whole_body_tracking`

### High-value files
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/events.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/exporter.py`
- `whole_body_tracking/scripts/rsl_rl/train.py`

### What the current environment already does well
- Uses Isaac Lab `ManagerBasedRLEnvCfg`.
- Uses RSL-RL PPO and a working training entrypoint.
- Uses a G1 robot config with realistic actuator limits, stiffness, damping, and armature.
- Uses default joint offsets and position actions, which is exactly what we want for standing.
- Already runs at a useful control rate:
  - `sim.dt = 0.005`
  - `decimation = 4`
  - effective policy rate = `1 / (0.005 * 4) = 50 Hz`
- Already has domain randomization for:
  - rigid-body friction
  - joint default position offsets
  - torso COM perturbation
  - external pushes
- Already has ONNX export support with metadata.

## Why This Is A Good Base

### 1. The robot config is already very close to what we need
`whole_body_tracking/whole_body_tracking/robots/g1.py` already defines:
- G1 URDF loading
- articulated body physics parameters
- per-joint actuator limits
- default standing-ish joint initialization
- automatically computed action scales based on `0.25 * effort / stiffness`

Important details already aligned with this repo:
- hip / knee / ankle default pose matches the current crouched stance
- arm default pose already uses elbows at `0.6`
- root spawn height is already near standing at `0.76`

This means the policy training environment is already much closer to your intended standby pose than a fresh import would be.

### 2. The training infrastructure is already solved
The project already includes:
- task registration
- PPO config
- Isaac Lab app launcher flow
- motion artifact download flow
- logging
- resume logic
- ONNX export helpers

For a standing task, we can delete complexity. We do not need to invent infrastructure.

### 3. The domain randomization patterns are reusable
The existing events in `tracking_env_cfg.py` are directly useful for standing:
- friction randomization
- joint default position calibration noise
- torso COM randomization
- push disturbances

These are exactly the kinds of randomization we want for a sim-to-real standby controller.

## Why It Is Not A Drop-In Solution

The current `whole_body_tracking` environment is designed for motion imitation, not stationary balance.

### Current tracking-specific assumptions
- It depends on `MotionCommandCfg`.
- It expects a motion file from WandB registry.
- It uses command-conditioned observations like anchor position/orientation relative to reference motion.
- Rewards are DeepMimic-style tracking terms:
  - anchor position / orientation error
  - relative body pose tracking
  - body linear / angular velocity tracking
- Terminations are based on deviation from the reference motion.
- Actions are whole-body over `[".*"]`, not limited to the standing-critical joints.

For standby, most of that needs to go away.

## Difficulty Assessment

### Overall difficulty: `easy-to-medium`

### Easy parts
- Reusing the G1 robot config
- Reusing Isaac Lab task structure
- Reusing PPO config and train script
- Reusing ONNX export flow
- Reusing randomization patterns
- Reusing 50 Hz control cadence

### Medium parts
- Rewriting the task from motion tracking to standing
- Choosing the right reduced action space
- Designing rewards that balance robustness and low jitter
- Making the trained policy deploy cleanly in ROS 2

### Hard parts
- Real-world validation with upper-body movement and XHand mass distribution
- Making observations line up exactly between Isaac Lab and ROS 2 deployment
- If needed, extending the current ROS controller stack to support a 15-joint standby policy cleanly

## Recommended New Task Design

### New task name
Create a new task rather than modifying the tracking task in place.

Suggested package/task path:
- `whole_body_tracking/.../tasks/standby/`
- task ID example: `Standby-Flat-G1XHand-v0`

### What to keep from `whole_body_tracking`
- `robots/g1.py` structure and actuator modeling
- RSL-RL PPO runner config style
- `train.py` script pattern
- exporter metadata pattern
- event randomization style
- Manager-based environment structure

### What to remove
- motion registry dependency
- `MotionCommandCfg`
- tracking-specific observations
- tracking-specific rewards
- tracking-specific terminations
- whole-body action space for standby

## Recommended Controlled Joints

Control only the lower body plus waist:

### Policy-controlled joints: 15
- `left_hip_pitch_joint`
- `left_hip_roll_joint`
- `left_hip_yaw_joint`
- `left_knee_joint`
- `left_ankle_pitch_joint`
- `left_ankle_roll_joint`
- `right_hip_pitch_joint`
- `right_hip_roll_joint`
- `right_hip_yaw_joint`
- `right_knee_joint`
- `right_ankle_pitch_joint`
- `right_ankle_roll_joint`
- `waist_yaw_joint`
- `waist_roll_joint`
- `waist_pitch_joint`

### Held fixed during standby training
Arms and hands should not be policy outputs in v1:
- arms fixed at default pose or lightly randomized per reset
- hands fixed open

This is the cleanest separation:
- standby policy = balance
- future manipulation policy = arms / hands

## Why 15 Joints Is Better Than Whole-Body Control
- Easier policy learning
- Smaller action space
- Less chance of degenerate arm-flailing solutions
- Cleaner deployment in ROS 2
- Better division of responsibility between balancing and manipulation

The waist should remain controlled because it helps counterbalance upper-body COM shifts.

## Recommended Observation Design

Use a small, deployment-friendly observation space that you can reproduce in ROS 2.

### Policy observations
- projected gravity in base frame: `3`
- base angular velocity: `3`
- lower-body joint position relative to default: `15`
- lower-body joint velocity: `15`
- previous action: `15`
- arm joint positions relative to default: `14`

### Total
- `65` dimensions

### Why this is good
- no dependency on privileged motion-tracking signals
- available in both sim and real robot
- explicitly lets the policy sense arm-induced COM shifts

## Recommended Action Design

Use position-offset actions around a nominal stance.

### Action space
- dimension: `15`
- action type: joint position target offset
- implementation style: same idea as `JointPositionActionCfg(..., use_default_offset=True)`

### Why this is the right choice
- matches the existing implicit actuator / PD style in Isaac Lab
- easier to stabilize than direct torque learning
- easier to map to your ROS controller

## Reward Design

The standing task should reward boring behavior:
- upright torso
- low drift
- low wobble
- low action jerk
- low joint velocity
- low torque
- feet planted
- no limit banging

### Suggested reward groups

#### Positive
- alive bonus
- upright reward
- feet-contact reward

#### Negative
- base XY velocity penalty
- base angular velocity penalty
- joint velocity penalty
- action rate penalty
- torque penalty
- joint limit penalty
- foot slip penalty
- deviation-from-default penalty

### Important design note
Do not overweight pose tracking. If default-pose tracking is too strong, the policy becomes another brittle PD servo.

## Domain Randomization To Reuse

Keep and adapt the existing `whole_body_tracking` randomization ideas:

### Keep
- friction randomization
- torso COM randomization
- push disturbances
- joint default offset randomization

### Add / adjust
- arm pose randomization at reset
- optional hand payload randomization
- slight pelvis height randomization at reset

### Why
This directly trains robustness to:
- arm position changes
- XHand mass effects
- floor variability
- calibration error
- pushes

## Concrete Gap Analysis Against `whole_body_tracking`

### Reusable without much change
- `robots/g1.py`
- `config/g1/agents/rsl_rl_ppo_cfg.py`
- `scripts/rsl_rl/train.py`
- `utils/exporter.py`
- event randomization patterns

### Needs partial rewrite
- env cfg
- observations
- rewards
- terminations
- task registration

### Should not be reused directly
- motion command machinery
- WandB motion registry requirement
- DeepMimic tracking reward definitions

## Proposed Implementation Steps In Isaac Lab

### Phase 1: fork the task
Create a new task beside tracking:
- `tasks/standby/`

Suggested files:
- `tasks/standby/standby_env_cfg.py`
- `tasks/standby/mdp/observations.py`
- `tasks/standby/mdp/rewards.py`
- `tasks/standby/mdp/events.py`
- `tasks/standby/mdp/terminations.py`
- `tasks/standby/config/g1/flat_env_cfg.py`
- `tasks/standby/config/g1/agents/rsl_rl_ppo_cfg.py`

### Phase 2: reuse the G1 robot config
Start from the existing G1 robot definition and only change what is needed for XHand / upper-body defaults.

### Phase 3: reduce the action space
Replace whole-body `joint_names=[".*"]` with the 15 standby joints.

### Phase 4: simplify the observation space
Remove all motion-anchor and reference-motion terms.

### Phase 5: replace rewards
Swap tracking rewards for standing rewards.

### Phase 6: replace terminations
Terminate on:
- pelvis too low
- bad orientation
- timeout

### Phase 7: keep randomization
Port over the useful randomization events and tune ranges conservatively first.

## How Easy Will Training Be?

### Expected training difficulty
Lower than motion tracking.

Reasons:
- no reference motion to match
- smaller action space
- simpler objective
- same robot and actuator config
- same 50 Hz policy cadence

### Likely training behavior
- early training: squat / wobble / ankle chatter
- mid training: learns quiet balancing around nominal crouch
- later training: learns push recovery and arm-pose compensation

### Likely convergence
If the environment is set up well, this should be much easier than full-body motion imitation.

## Real Deployment Plan

This is where `STANDBY_POLICY_PLAN.md` and the existing ROS 2 stack come together.

### Training output
Train in Isaac Lab, export a compact ONNX actor.

### Deployment target in this repo
Use the ONNX in a ROS 2 controller similar to:
- `xhand_manip_controller`

### Real controller behavior
At runtime the standby policy should:
- read IMU orientation / angular velocity
- read 15 lower-body joint positions / velocities
- read arm joint positions
- compute the 65-dim observation
- run ONNX at 50 Hz
- output 15 joint target offsets
- send targets into the existing low-level PD pipeline

### Real-life deployment structure
- policy loop: `50 Hz`
- hardware / PD loop: `500 Hz`
- standby policy controls:
  - legs
  - waist
- manipulation stack controls:
  - arms
  - hands

## Why Deployment Should Be Reasonably Straightforward

Because `whole_body_tracking` already exports ONNX with metadata patterns, and this repo already has ONNX-based ROS controller infrastructure, the sim-to-deployment bridge is conceptually simple:
- train in Isaac Lab
- export ONNX
- reproduce observation ordering in ROS
- apply action scaling and default offsets

The main challenge is not exporting the model. The main challenge is making sure the ROS observation vector exactly matches the Isaac Lab observation vector.

## Biggest Real-World Risks

### 1. Observation mismatch
If the ROS observation order, normalization, or frame conventions differ from training, the policy will fail immediately.

### 2. Upper-body mass mismatch
The G1 config inside `whole_body_tracking` is stock G1-oriented. Your robot has XHands and different arm inertia distribution, so the Isaac asset may need adjustment.

### 3. Action scaling mismatch
If Isaac Lab action scales differ from what the ROS controller expects, the deployed policy can become too weak or too aggressive.

### 4. Contact / friction mismatch
Standing success is sensitive to friction modeling. Randomization helps, but real-floor testing still matters.

## Recommended Modifications Before Training

### In `whole_body_tracking`
- create a new standby task instead of reusing tracking directly
- adapt G1 asset to reflect XHand mass if possible
- fix arm default pose to match this repo
- keep policy rate at 50 Hz
- keep domain randomization, but simplify first

### In this repo
- keep `STANDBY_POLICY_PLAN.md` as the deployment-oriented spec
- use the new Isaac Lab file as the training-oriented spec
- later build a dedicated `StandbyPolicyController`

## Suggested Training Strategy

### Stage 1
Train standing with fixed arms, no payload randomization, mild pushes.

### Stage 2
Add arm pose randomization at reset.

### Stage 3
Add torso COM randomization and stronger pushes.

### Stage 4
Add payload randomization if you want manipulation-ready balance.

## Recommended Final Architecture

```text
Isaac Lab standby task
  -> PPO training
  -> ONNX export
  -> ROS 2 StandbyPolicyController
  -> 15-joint balance outputs
  -> existing low-level PD / hardware loop
```

Later:

```text
Standby policy: legs + waist
Manipulation policy: arms + hands
Shared state estimator / IMU / joint state pipeline
```

## Final Recommendation

Yes, it is worth using `whole_body_tracking` as the Isaac Lab base.

Best long-term path:
- ignore the nested repo from the parent workspace
- fork its tracking task into a new standby task
- keep its G1 robot config, PPO setup, exporter, and randomization style
- remove motion imitation dependencies
- train a 15-joint standing policy
- deploy the ONNX back into this ROS 2 workspace

## Search Prompts For Future GPT / Web Research

If you want to hand this file to another GPT later, these are the highest-value follow-up questions:

- How to create a new Isaac Lab `ManagerBasedRLEnv` task by forking an existing task?
- Best practice for exporting RSL-RL PPO policies from Isaac Lab to ONNX for ROS deployment?
- How to validate observation ordering consistency between Isaac Lab and ROS 2?
- How to model added hand mass / inertia in Isaac Lab G1 assets?
- Best standing rewards for humanoid balance with fixed upper body?
- How to split balance policy and manipulation policy on humanoids?
