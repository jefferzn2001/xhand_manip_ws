import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PKG_NAME = "g1_xhand_description"


def _get_config_path():
    """Get absolute path to controllers.yaml."""
    pkg_share = get_package_share_directory(PKG_NAME)
    return os.path.join(pkg_share, "config", "controllers.yaml")


def _get_controller_names(config_path):
    """Read controller names from controllers.yaml."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    cm_params = data.get("controller_manager", {}).get("ros__parameters", {})
    skip = {"update_rate", "use_sim_time"}
    return [k for k in cm_params if k not in skip and isinstance(cm_params[k], dict)]


def _generate_temp_config(config_path, kv_overrides):
    """
    Create a temp copy of controllers.yaml with key-value overrides applied.

    Supports dotted keys like 'walking_controller.policy.path'.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    for dotted_key, value in kv_overrides:
        parts = dotted_key.split(".")
        node = data
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if "ros__parameters" in node[part] and parts.index(part) == 0:
                node = node[part]["ros__parameters"]
            else:
                node = node[part]
        node[parts[-1]] = value

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="controllers_", delete=False
    )
    yaml.dump(data, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def _resolve_policy_paths(config_path):
    """Find relative policy paths in YAML and make them absolute."""
    pkg_share = get_package_share_directory(PKG_NAME)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    overrides = []
    for ctrl_name, ctrl_cfg in data.items():
        if not isinstance(ctrl_cfg, dict):
            continue
        params = ctrl_cfg.get("ros__parameters", {})
        policy = params.get("policy", {})
        if isinstance(policy, dict) and "path" in policy:
            rel_path = policy["path"]
            if not os.path.isabs(rel_path):
                abs_path = os.path.join(pkg_share, rel_path)
                overrides.append((f"{ctrl_name}.policy.path", abs_path))
    return overrides


def _control_spawner(controllers, param_file, inactive=False):
    """Create a controller spawner node."""
    args = list(controllers)
    args += ["--param-file", param_file]
    if inactive:
        args.append("--inactive")
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=args,
        output="screen",
    )


def setup_controllers(context):
    """Configure and spawn controllers."""
    policy_path_value = LaunchConfiguration("policy_path").perform(context)
    config_path = _get_config_path()

    kv_pairs = _resolve_policy_paths(config_path)
    if policy_path_value:
        abs_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(policy_path_value))
        )
        kv_pairs.append(("walking_controller.policy.path", abs_path))

    temp_config = _generate_temp_config(config_path, kv_pairs)

    set_yaml = SetLaunchConfiguration(name="controllers_yaml", value=temp_config)

    all_controllers = _get_controller_names(config_path)
    active_list = ["state_estimator", "walking_controller"]
    inactive_list = [c for c in all_controllers if c not in active_list]

    param_file = LaunchConfiguration("controllers_yaml")
    active = _control_spawner(active_list, param_file=param_file)
    inactive = _control_spawner(inactive_list, param_file=param_file, inactive=True)

    return [set_yaml, active, inactive]


def generate_launch_description():
    """Launch G1+XHand in MuJoCo simulation."""
    robot_description_command = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare(PKG_NAME),
                    "urdf",
                    "g1_xhand",
                    "robot.xacro",
                ]
            ),
            " ",
            "simulation:=mujoco",
        ]
    )
    robot_description = {"robot_description": robot_description_command}

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"publish_frequency": 500.0, "use_sim_time": True},
        ],
    )

    mujoco_simulator = Node(
        package="mujoco_sim_ros2",
        executable="mujoco_sim",
        parameters=[
            {
                "model_package": PKG_NAME,
                "model_file": "/mjcf/g1_xhand.xml",
                "physics_plugins": [
                    "mujoco_ros2_control::MujocoRos2ControlPlugin"
                ],
                "use_sim_time": True,
            },
            robot_description,
            LaunchConfiguration("controllers_yaml"),
        ],
        output="screen",
    )

    controllers_setup = OpaqueFunction(function=setup_controllers)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "policy_path",
                default_value="",
                description="Absolute path to ONNX policy file",
            ),
            controllers_setup,
            mujoco_simulator,
            robot_state_publisher,
        ]
    )
