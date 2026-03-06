import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from legged_bringup.launch_utils import (
    get_controller_names,
    generate_temp_config,
    resolve_policy_paths,
    control_spawner,
)


def setup_controllers(context):
    """Configure controllers from g1_xhand_description config."""
    policy_path_value = LaunchConfiguration("policy_path").perform(context)

    controllers_config_path = "config/controllers.yaml"
    pkg_name = "g1_xhand_description"

    kv_pairs = resolve_policy_paths(controllers_config_path, pkg_name)
    if policy_path_value:
        abs_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(policy_path_value))
        )
        kv_pairs.append(("walking_controller.policy.path", abs_path))
    temp_controllers_config_path = generate_temp_config(
        controllers_config_path, pkg_name, kv_pairs
    )

    set_controllers_yaml = SetLaunchConfiguration(
        name="controllers_yaml", value=temp_controllers_config_path
    )

    all_controllers = get_controller_names(controllers_config_path, pkg_name)
    active_list = ["state_estimator", "walking_controller"]
    inactive_list = [c for c in all_controllers if c not in active_list]
    param_file = LaunchConfiguration("controllers_yaml")
    active_spawner = control_spawner(active_list, param_file=param_file)
    inactive_spawner = control_spawner(
        inactive_list, param_file=param_file, inactive=True
    )
    return [set_controllers_yaml, active_spawner, inactive_spawner]


def generate_launch_description():
    """Launch G1+XHand in MuJoCo simulation."""
    # URDF from g1_xhand_description (combined G1 + XHand)
    robot_description_command = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_xhand_description"),
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

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"publish_frequency": 500.0, "use_sim_time": True},
        ],
    )

    # MuJoCo uses the MJCF from g1_xhand_description
    mujoco_simulator = Node(
        package="mujoco_sim_ros2",
        executable="mujoco_sim",
        parameters=[
            {
                "model_package": "g1_xhand_description",
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

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    teleop = PathJoinSubstitution(
        [FindPackageShare("unitree_bringup"), "launch", "teleop.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_type", default_value="g1"),
            DeclareLaunchArgument(
                "policy_path",
                default_value="",
                description="Absolute path to ONNX policy file",
            ),
            controllers_opaque_func,
            mujoco_simulator,
            node_robot_state_publisher,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(teleop),
                launch_arguments={
                    "robot_type": LaunchConfiguration("robot_type")
                }.items(),
            ),
        ]
    )
