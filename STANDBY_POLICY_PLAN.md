# Standby Policy Plan — RL Standing Controller for G1 + XHand

> Training framework: **Isaac Lab** (ManagerBasedRLEnv + RSL-RL PPO)

## Problem

The current `StandbyController` is a fixed-pose PD servo (`controllers.yaml` `default_position`).
It drives all 53 joints to a single target with constant gains. It has no concept of balance
or center-of-mass tracking. When the robot spawns or the arms move, the COM shifts and the
robot falls because the controller can't compensate.

## Goal

Train an RL **standing policy** that actively balances the lower body, replacing the
fixed PD standby controller. The policy must:

- Keep the robot upright and stationary
- Be robust to upper-body movements (arms reaching, holding objects)
- Handle external perturbations (pushes)
- Generalize across payload variation (XHands weigh ~1.17 kg each)
- Export to ONNX for deployment in the existing ros2_control stack

---

## Robot Specification

| Property | Value |
|----------|-------|
| Robot | Unitree G1 + XHand dexterous hands |
| URDF | `urdf/g1_xhand/robot.xacro` (convert to USD for Isaac Lab) |
| MJCF (reference) | `mjcf/g1_xhand.xml` |
| Total actuated joints | 53 |
| Standing height (pelvis) | ~0.74 m (crouched nominal) |
| Total mass | ~35 kg (G1) + ~2.34 kg (two XHands) |
| Policy frequency | 50 Hz |
| Sim dt | 1/500 s (decimation = 10) |

---

## Joint Decomposition

### Controlled by standing policy (15 joints)

These are the only joints the policy outputs actions for.

```
LEFT LEG (6):
  left_hip_pitch_joint        axis: Y   range: [-2.5307, 2.8798]   torque: 88 Nm
  left_hip_roll_joint         axis: X   range: [-0.5236, 2.9671]   torque: 139 Nm
  left_hip_yaw_joint          axis: Z   range: [-2.7576, 2.7576]   torque: 88 Nm
  left_knee_joint             axis: Y   range: [-0.0873, 2.8798]   torque: 139 Nm
  left_ankle_pitch_joint      axis: Y   range: [-0.8727, 0.5236]   torque: 50 Nm
  left_ankle_roll_joint       axis: X   range: [-0.2618, 0.2618]   torque: 50 Nm

RIGHT LEG (6):
  right_hip_pitch_joint       axis: Y   range: [-2.5307, 2.8798]   torque: 88 Nm
  right_hip_roll_joint        axis: X   range: [-2.9671, 0.5236]   torque: 139 Nm
  right_hip_yaw_joint         axis: Z   range: [-2.7576, 2.7576]   torque: 88 Nm
  right_knee_joint            axis: Y   range: [-0.0873, 2.8798]   torque: 139 Nm
  right_ankle_pitch_joint     axis: Y   range: [-0.8727, 0.5236]   torque: 50 Nm
  right_ankle_roll_joint      axis: X   range: [-0.2618, 0.2618]   torque: 50 Nm

WAIST (3):
  waist_yaw_joint             axis: Z   range: [-2.618, 2.618]     torque: 88 Nm
  waist_roll_joint            axis: X   range: [-0.52, 0.52]       torque: 50 Nm
  waist_pitch_joint           axis: Y   range: [-0.52, 0.52]       torque: 50 Nm
```

### NOT controlled by standing policy (38 joints)

Held at fixed default positions during training. Later driven by a separate manipulation controller.

```
LEFT ARM (7):   shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
RIGHT ARM (7):  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
LEFT HAND (12): thumb(3), index(3), mid(2), ring(2), pinky(2)
RIGHT HAND (12): thumb(3), index(3), mid(2), ring(2), pinky(2)
```

The waist is included in the policy because it is critical for balance compensation when
the arms move or hold objects.

---

## Isaac Lab Environment Config

### Asset Import

Convert the URDF to USD for Isaac Lab:

```bash
# From Isaac Lab root
python scripts/tools/convert_urdf.py \
  /path/to/g1_xhand_description/urdf/g1_xhand/robot.xacro \
  /path/to/output/g1_xhand.usd \
  --merge-fixed-joints \
  --make-instanceable
```

Then define the robot config:

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

G1_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/g1_xhand.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            # Legs
            "left_hip_pitch_joint": -0.312,
            "left_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "right_hip_pitch_joint": -0.312,
            "right_knee_joint": 0.669,
            "right_ankle_pitch_joint": -0.363,
            # Arms
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_elbow_joint": 0.6,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_elbow_joint": 0.6,
            # Everything else defaults to 0
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip_pitch_joint", ".*hip_roll_joint", ".*hip_yaw_joint",
                ".*knee_joint",
            ],
            stiffness={
                ".*hip_pitch_joint": 350.0,
                ".*hip_roll_joint": 200.0,
                ".*hip_yaw_joint": 200.0,
                ".*knee_joint": 300.0,
            },
            damping={
                ".*hip_pitch_joint": 5.0,
                ".*hip_roll_joint": 5.0,
                ".*hip_yaw_joint": 5.0,
                ".*knee_joint": 10.0,
            },
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch_joint", ".*ankle_roll_joint"],
            stiffness={".*ankle_pitch_joint": 300.0, ".*ankle_roll_joint": 150.0},
            damping={".*ankle_pitch_joint": 5.0, ".*ankle_roll_joint": 5.0},
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=200.0,
            damping=5.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_.*_joint", ".*elbow_joint", ".*wrist_.*_joint",
            ],
            stiffness=40.0,
            damping=5.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*hand_.*_joint"],
            stiffness=2.0,
            damping=0.1,
        ),
    },
)
```

### Scene

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

@configclass
class G1StandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    robot = G1_XHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )
```

---

## Observation Space (65-dim)

| # | Observation | Dim | Isaac Lab function | Notes |
|---|-------------|-----|--------------------|-------|
| 1 | Projected gravity in body frame | 3 | `mdp.projected_gravity` | Which way is "up" relative to torso |
| 2 | Base angular velocity (body frame) | 3 | `mdp.base_ang_vel` | Roll/pitch/yaw rates |
| 3 | Lower-body joint positions | 15 | `mdp.joint_pos_rel` | 15 controlled joints, relative to default |
| 4 | Lower-body joint velocities | 15 | `mdp.joint_vel_rel` | 15 controlled joints |
| 5 | Previous action | 15 | `mdp.last_action` | For temporal smoothness |
| 6 | Upper-body joint positions | 14 | `mdp.joint_pos_rel` | Arms only (7L + 7R), sense COM disturbance |
| | **Total** | **65** | | |

```python
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        lower_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
        )
        lower_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

---

## Action Space (15-dim)

Actions are **position offsets** from the nominal standing pose. Isaac Lab's
`JointPositionActionCfg` with `use_default_offset=True` handles this natively.

```python
from isaaclab.managers import ActionTermCfg

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_JOINTS,
        scale={
            ".*hip_pitch_joint": 0.4,
            ".*hip_roll_joint": 0.2,
            ".*hip_yaw_joint": 0.2,
            ".*knee_joint": 0.4,
            ".*ankle_pitch_joint": 0.3,
            ".*ankle_roll_joint": 0.15,
            "waist_yaw_joint": 0.3,
            "waist_roll_joint": 0.2,
            "waist_pitch_joint": 0.2,
        },
        use_default_offset=True,  # action = default_pos + output * scale
    )
```

### Nominal standing pose (default joint positions from controllers.yaml)

```python
default_standing_pos = {
    "left_hip_pitch_joint":   -0.312,
    "left_hip_roll_joint":     0.0,
    "left_hip_yaw_joint":      0.0,
    "left_knee_joint":         0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint":   0.0,
    "right_hip_pitch_joint":  -0.312,
    "right_hip_roll_joint":    0.0,
    "right_hip_yaw_joint":     0.0,
    "right_knee_joint":        0.669,
    "right_ankle_pitch_joint":-0.363,
    "right_ankle_roll_joint":  0.0,
    "waist_yaw_joint":         0.0,
    "waist_roll_joint":        0.0,
    "waist_pitch_joint":       0.0,
}
```

### PD gains (set in actuator config above)

| Group | kp | kd |
|-------|----|----|
| hip_pitch | 350 | 5 |
| hip_roll | 200 | 5 |
| hip_yaw | 200 | 5 |
| knee | 300 | 10 |
| ankle_pitch | 300 | 5 |
| ankle_roll | 150 | 5 |
| waist | 200 | 5 |

---

## Reward Function

```python
from isaaclab.managers import RewardTermCfg as RewTerm

@configclass
class RewardsCfg:
    # --- Positive (what we want) ---

    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    upright = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=1.0,
        params={"threshold": 0.9},
    )

    zero_xy_velocity = RewTerm(
        func=mdp.base_lin_vel_l2,  # custom: exp(-2 * (vx^2 + vy^2))
        weight=-0.5,               # penalty form OR write custom reward
    )

    default_pose = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )

    # --- Penalties (what we don't want) ---

    ang_vel = RewTerm(func=mdp.base_ang_vel_l2, weight=-0.05)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
    )
```

### Reward summary

| Name | Type | Weight | Purpose |
|------|------|--------|---------|
| alive | + | 2.0 | Survive |
| upright | + | 1.0 | Torso vertical |
| zero_xy_velocity | - | -0.5 | Stay in place |
| default_pose | - | -0.3 | Soft preference to nominal stance |
| ang_vel | - | -0.05 | No wobble |
| joint_vel | - | -0.01 | Smooth motion |
| action_rate | - | -0.05 | No jitter |
| joint_torque | - | -0.001 | Energy efficient |
| joint_pos_limits | - | -1.0 | Stay away from limits |

> **Note**: Some of these terms (e.g. `base_lin_vel_l2`, `joint_deviation_l1`)
> may need custom implementations in your `mdp/` module depending on your
> Isaac Lab version. Check `isaaclab.envs.mdp` for available built-in terms
> and write custom ones for anything missing (foot contact, foot slip, etc.).

---

## Terminations

```python
from isaaclab.managers import TerminationTermCfg as DoneTerm

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    pelvis_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45},
    )

    # Custom: terminate if torso tilt > 60 degrees
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 1.05},  # radians
    # )
```

---

## Events (Reset + Domain Randomization)

```python
from isaaclab.managers import EventTermCfg as EventTerm

@configclass
class EventCfg:
    # --- Reset events ---
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"z": (0.72, 0.76)},
            "velocity_range": {},
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # --- Randomize arm poses each reset (COM disturbance) ---
    randomize_arms = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*shoulder_.*_joint", ".*elbow_joint",
            ]),
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    # --- Interval events (during episode) ---

    # Random pushes every 2-5 seconds
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )

    # Randomize friction
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
        },
    )

    # Randomize link masses (+/- 10%)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
```

---

## Full Environment Config

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

@configclass
class G1StandEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1StandSceneCfg = G1StandSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 10            # 500 Hz sim / 10 = 50 Hz policy
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 500.0       # 500 Hz physics, matches real PD loop
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
```

---

## Training

### RSL-RL PPO config

```python
# rsl_rl_cfg.py
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg

G1_STAND_RSL_RL_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=5000,
    experiment_name="g1_xhand_stand",
    policy=RslRlPpoActorCriticCfg(
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        clip_param=0.2,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        desired_kl=0.01,
        schedule="adaptive",
    ),
)
```

### Launch training

```bash
cd IsaacLab

python source/isaaclab_rl/scripts/rsl_rl/train.py \
  --task G1-XHand-Stand-v0 \
  --num_envs 4096 \
  --headless
```

### Training curriculum (manual)

| Phase | Iterations | What changes |
|-------|------------|-------------|
| 1 | 0-500 | No perturbations, fixed arm pose, no mass randomization |
| 2 | 500-2000 | Enable `randomize_arms`, add small pushes |
| 3 | 2000-4000 | Enable `push_robot`, `randomize_mass`, `randomize_friction` |
| 4 | 4000+ | Increase push velocity range to (-1.0, 1.0) |

---

## Export to ONNX

After training, export the actor network:

```python
import torch
from rsl_rl.modules import ActorCritic

# Load checkpoint
model = ActorCritic(num_obs=65, num_actions=15, actor_hidden_dims=[256, 128, 128], ...)
model.load_state_dict(torch.load("logs/g1_xhand_stand/model_5000.pt"))
model.eval()

dummy_obs = torch.zeros(1, 65)
torch.onnx.export(
    model.actor,
    dummy_obs,
    "standby_policy.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=11,
)
```

Copy `standby_policy.onnx` to `g1_xhand_description/config/` for deployment.

---

## Deployment in ROS 2

The existing `XhandManipController` extends `OnnxController` and loads ONNX policies.
Build a similar `StandbyPolicyController` that:

1. Reads the 65-dim observation from joint states + IMU
2. Runs the ONNX policy at 50 Hz
3. Outputs 15 target joint positions (`default_pos + action * scale`)
4. The hardware interface PD loop runs at 500 Hz

```yaml
# controllers.yaml (after training)
standby_controller:
  ros__parameters:
    type: standby_policy_controller/StandbyPolicyController
    update_rate: 50
    policy:
      path: "config/standby_policy.onnx"
```

### Long-term architecture

```
              Manipulation Planner
                  |            |
       Standing Policy    Manipulation Policy
       (RL, 15 joints)   (RL/IK, 38 joints)
       legs + waist       arms + hands
                  |            |
              PD Torque Loop (500 Hz)
                  |
           MuJoCo / Real HW
```

The standing policy observes arm positions so it compensates for COM shifts.
The manipulation policy assumes the legs keep the robot balanced. Both run concurrently.

---

## Custom MDP Terms to Implement

These reward/observation terms may not exist in Isaac Lab's built-in `mdp` module
and will need custom implementations in your task's `mdp/` folder:

| Term | Type | Notes |
|------|------|-------|
| `foot_contact_reward` | Reward | `+0.25` per foot with contact force > threshold |
| `foot_slip_penalty` | Reward | `-||v_foot_xy||^2` when foot is in contact |
| `zero_xy_velocity_reward` | Reward | `exp(-2 * (vx^2 + vy^2))` on base |
| `bad_orientation` | Termination | `acos(proj_gravity_z) > limit_angle` |
| `projected_gravity` | Observation | `R_body^T @ [0,0,-1]` (may already exist) |

---

## Files Reference

| File | Role |
|------|------|
| `mjcf/g1_xhand.xml` | MuJoCo model (reference for joint specs, contacts) |
| `urdf/g1_xhand/robot.xacro` | Source URDF (convert to USD for Isaac Lab) |
| `config/controllers.yaml` | PD gains, default positions, joint names |
| `src/xhand_manip_controller/` | Existing ONNX controller (reference for ROS deployment) |

---

## Summary

| Item | Value |
|------|-------|
| Framework | Isaac Lab (ManagerBasedRLEnv) |
| Algorithm | RSL-RL PPO |
| Controlled joints | 15 (6 left leg + 6 right leg + 3 waist) |
| Observation dim | 65 |
| Action dim | 15 (position offsets from nominal) |
| Policy frequency | 50 Hz (decimation=10 at 500 Hz sim) |
| Sim environments | 4096 |
| Network | MLP [256, 128, 128] ELU |
| Export format | ONNX (opset 11) |
