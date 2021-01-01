cmake_minimum_required(VERSION 2.8.3)
project(encoder_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  image_geometry
  duckietown_msgs
  tf
  std_msgs
  geometry_msgs
)

catkin_package()
