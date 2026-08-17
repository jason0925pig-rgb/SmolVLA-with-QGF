"""Real-robot SmolVLA policy server with one visual IQL critic for QGF.

This module has no robot SDK, publisher, service or power-control code.  It
only changes the normalized SmolVLA action chunk inside the existing gRPC
policy server; the attended ROS client and its safety gates remain unchanged.
"""

import os
from pathlib import Path

import draccus

from guided_action_flow.critics.checkpoint import load_action_chunk_critic
from guided_action_flow.guidance.qgf import QGuidanceConfig
from guided_action_flow.policies.smolvla_qgf import (
    SmolVLAVisualCriticAdapter,
    install_smolvla_qgf,
)
from lerobot.async_inference.configs import PolicyServerConfig

from lerobot_robot_armstrong_ros2.policy_server_telemetry import TelemetryPolicyServer


def _positive_env_float(name: str) -> float:
    try:
        value = float(os.environ[name])
    except KeyError as exc:
        raise RuntimeError(f"{name} must be set for the QGF policy server.") from exc
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a finite positive number.") from exc
    if value <= 0.0:
        raise RuntimeError(f"{name} must be positive, got {value}.")
    return value


class QGFPolicyServer(TelemetryPolicyServer):
    """Install exactly one visual Q critic after the policy is loaded."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        critic_path = Path(os.environ.get("SMOLVLA_QGF_CRITIC_PATH", ""))
        if not critic_path.is_file():
            raise RuntimeError(
                "SMOLVLA_QGF_CRITIC_PATH must point to the deployed "
                f"single-critic checkpoint; got {critic_path}."
            )
        beta = _positive_env_float("SMOLVLA_QGF_BETA")
        grad_clip_norm = _positive_env_float("SMOLVLA_QGF_GRAD_CLIP_NORM")
        critic, metadata = load_action_chunk_critic(critic_path, device=self.device)
        if metadata.get("critic_arch") != "visual_transformer":
            raise RuntimeError(
                "The real-robot QGF server requires a visual_transformer critic, "
                f"not {metadata.get('critic_arch')!r}."
            )
        if int(metadata["critic_config"]["action_dim"]) != 8:
            raise RuntimeError("The deployed Armstrong critic must use eight action channels.")
        adapter = SmolVLAVisualCriticAdapter(critic)
        install_smolvla_qgf(
            self.policy,
            critic=adapter,
            config=QGuidanceConfig(
                beta=beta,
                grad_clip_norm=grad_clip_norm,
                uncertainty_scale=0.0,
                min_gate=0.0,
            ),
            critic_action_dim=8,
        )
        self.logger.info(
            "QGF single-critic guidance installed: "
            f"checkpoint={critic_path}; beta={beta:.8g}; coefficient=1/beta={1.0 / beta:.8g}; "
            f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled"
        )
        return response


@draccus.wrap()
def serve(config: PolicyServerConfig) -> None:
    server_impl = QGFPolicyServer(config)
    from concurrent import futures

    import grpc
    from lerobot.transport import services_pb2_grpc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()
    server_impl.logger.info("QGF policy server started; telemetry remains on /smolvla/normalized_action_chunk")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server_impl.logger.info("QGF policy server interrupted")
    finally:
        server.stop(grace=1.0)
        server_impl.stop()


if __name__ == "__main__":
    serve()
