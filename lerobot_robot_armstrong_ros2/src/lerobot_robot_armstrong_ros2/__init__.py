"""Armstrong ROS 2 plugin discovered by LeRobot's third-party loader.

The public classes are loaded lazily so the pure-Python safety guard and its
tests remain usable on development machines that do not have LeRobot/ROS2.
"""

from typing import Any

__all__ = ["ArmstrongRos2", "ArmstrongRos2Config"]


def __getattr__(name: str) -> Any:
    if name == "ArmstrongRos2Config":
        from .configuration_armstrong_ros2 import ArmstrongRos2Config

        return ArmstrongRos2Config
    if name == "ArmstrongRos2":
        from .armstrongros2 import ArmstrongRos2

        return ArmstrongRos2
    raise AttributeError(name)
