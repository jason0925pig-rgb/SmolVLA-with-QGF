"""LeRobot policy server that passively publishes pre-postprocessor chunks."""

from __future__ import annotations

import threading
from concurrent import futures

import draccus
import grpc
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


ACTION_NAMES = tuple(f"right_joint{i}" for i in range(1, 8)) + (
    "right_gripper_closed",
)


class TelemetryPolicyServer(PolicyServer):
    """Expose normalized flow-policy chunks without changing inference output."""

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        if not rclpy.ok():
            rclpy.init(args=None)
        self._telemetry_node = Node("smolvla_policy_chunk_telemetry")
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._normalized_chunk_pub = self._telemetry_node.create_publisher(
            JointTrajectory, "/smolvla/normalized_action_chunk", qos
        )
        self._telemetry_executor = SingleThreadedExecutor()
        self._telemetry_executor.add_node(self._telemetry_node)
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_executor.spin, daemon=True
        )
        self._telemetry_thread.start()
        self._normalized_chunk = None
        self._chunk_lock = threading.Lock()

    def _get_action_chunk(self, observation):
        chunk = super()._get_action_chunk(observation)
        with self._chunk_lock:
            self._normalized_chunk = chunk.detach().to("cpu").clone()
        return chunk

    def _predict_action_chunk(self, observation_t):
        actions = super()._predict_action_chunk(observation_t)
        with self._chunk_lock:
            chunk = self._normalized_chunk
        if chunk is not None:
            values = chunk.squeeze(0)
            message = JointTrajectory()
            message.header.stamp = self._telemetry_node.get_clock().now().to_msg()
            message.header.frame_id = str(observation_t.get_timestep())
            message.joint_names = list(ACTION_NAMES)
            for row in values:
                point = JointTrajectoryPoint()
                point.positions = [float(value) for value in row.tolist()]
                message.points.append(point)
            self._normalized_chunk_pub.publish(message)
        return actions

    def stop(self):
        super().stop()
        self._telemetry_executor.shutdown(timeout_sec=2.0)
        self._telemetry_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@draccus.wrap()
def serve(config: PolicyServerConfig) -> None:
    server_impl = TelemetryPolicyServer(config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()
    server_impl.logger.info(
        "Telemetry policy server started; normalized chunks publish on "
        "/smolvla/normalized_action_chunk"
    )
    try:
        server.wait_for_termination()
    finally:
        server_impl.stop()


if __name__ == "__main__":
    serve()
