# LiDAR Localization ROS Node

## Overview

A ROS node for 2D LiDAR-based robot localization using `nav_msgs/OccupancyGrid` and `sensor_msgs/LaserScan`. Estimates robot pose, publishes map to odom transform via TF2, and integrates with `move_base`.

## Features

* Crops map around occupied cells
* Aligns LiDAR scans with gradient-based scoring
* Handles initial pose from `/initialpose` topic
* Checks pose convergence
* Clears costmaps via `/move_base/clear_costmaps`

## Prerequisites

* ROS (Noetic or Melodic)
* OpenCV, TF2
* ROS packages:
  * `nav_msgs`
  * `sensor_msgs`
  * `geometry_msgs`
  * `tf2`
  * `tf2_ros`
  * `cv_bridge`
  * `std_srvs`
* LiDAR and map server
