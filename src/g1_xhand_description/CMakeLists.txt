cmake_minimum_required(VERSION 3.8)
project(g1_xhand_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf mjcf meshes config launch scripts
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/combine_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
