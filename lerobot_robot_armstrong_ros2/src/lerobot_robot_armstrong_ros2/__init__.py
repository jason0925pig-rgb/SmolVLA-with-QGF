"""Armstrong ROS 2 plugin discovered by LeRobot's third-party loader."""

# LeRobot 0.4.4 discovers third-party robot types by inspecting classes that
# are present directly in the imported package module.  Lazy __getattr__
# exports are invisible to that scanner and leave the draccus robot choices
# empty, so these two public plugin classes must be imported eagerly.
from .configuration_armstrong_ros2 import ArmstrongRos2Config
from .armstrongros2 import ArmstrongRos2

__all__ = ["ArmstrongRos2", "ArmstrongRos2Config"]
