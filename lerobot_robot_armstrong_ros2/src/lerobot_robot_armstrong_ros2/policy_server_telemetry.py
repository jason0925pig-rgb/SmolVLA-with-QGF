"""LeRobot policy server that publishes chunks with selectable AMP inference."""

import os
import threading
from concurrent import futures

import draccus
import grpc
import rclpy
import torch
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


ACTION_NAMES = tuple(f"right_joint{i}" for i in range(1, 8)) + (
    "right_gripper_closed",
)


def _resolve_inference_dtype() -> torch.dtype | None:
    """Map the attended inference setting to a CUDA autocast dtype."""
    mode = os.environ.get("SMOLVLA_INFERENCE_DTYPE", "fp32").strip().lower()
    dtypes = {
        "fp32": None,
        "float32": None,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if mode not in dtypes:
        raise ValueError(
            "SMOLVLA_INFERENCE_DTYPE must be fp32, fp16, or bf16; "
            f"got {mode!r}."
        )
    return dtypes[mode]


class TelemetryPolicyServer(PolicyServer):
    """Expose normalized flow-policy chunks without changing inference output."""

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        self._inference_dtype = _resolve_inference_dtype()
        if self._inference_dtype is not None and not torch.cuda.is_available():
            raise RuntimeError("mixed-precision policy inference requires CUDA")
        self.logger.info(
            "POLICY_INFERENCE_PRECISION mode=%s autocast=%s",
            os.environ.get("SMOLVLA_INFERENCE_DTYPE", "fp32").strip().lower(),
            self._inference_dtype is not None,
        )
        if not rclpy.ok():
            # Keep SIGINT/SIGTERM under the gRPC main process.  Otherwise
            # rclpy consumes the signal, shuts down only the ROS executor and
            # leaves server.wait_for_termination() alive as a stale process.
            rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
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
        if self._inference_dtype is None:
            chunk = super()._get_action_chunk(observation)
        else:
            # Keep model weights and numerically sensitive operations safe while
            # routing supported CUDA kernels through FP16/BF16 Tensor Cores.
            with torch.autocast(device_type="cuda", dtype=self._inference_dtype):
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
    except KeyboardInterrupt:
        server_impl.logger.info("Telemetry policy server interrupted")
    finally:
        server.stop(grace=1.0)
        server_impl.stop()


if __name__ == "__main__":
    serve()
