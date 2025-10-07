cmake_minimum_required(VERSION 2.6)
PROJECT(  tracker_camera )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Pose )
