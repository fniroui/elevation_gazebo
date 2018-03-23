cmake_minimum_required(VERSION 2.8.3)
project(elevation_gazebo)

find_package(catkin REQUIRED COMPONENTS roslaunch gazebo_ros)

catkin_package()

roslaunch_add_file_check(launch)

install(
  DIRECTORY launch worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
