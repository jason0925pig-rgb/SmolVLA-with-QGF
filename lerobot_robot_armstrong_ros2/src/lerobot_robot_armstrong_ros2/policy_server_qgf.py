"""Real-robot SmolVLA policy server with one visual IQL critic for QGF.

This module has no robot SDK, publisher, service or power-control code.  It
only changes the normalized SmolVLA action chunk inside the existing gRPC
policy server; the attended ROS client and its safety gates remain unchanged.
"""

import os
import torch
from pathlib import Path

import draccus

from guided_action_flow.critics.checkpoint import load_action_chunk_critic
from guided_action_flow.guidance.qgf import GUIDANCE_MODES, QGuidanceConfig
from guided_action_flow.policies.smolvla_qgf import (
    SmolVLAStateActionCriticAdapter,
    SmolVLAVisualActionCriticAdapter,
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
    """Install exactly one visual or state/action Q critic after policy load."""

    _qgf_chunk_counter = 0

    def _get_action_chunk(self, observation):
        # Preserve the deployed ablation latency instrumentation.  The
        # no-vision input ablation must be compared with measured per-chunk
        # latency rather than silently dropping the existing timing path.
        import time as _time

        t0 = _time.perf_counter()
        chunk = super()._get_action_chunk(observation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        QGFPolicyServer._qgf_chunk_counter += 1
        self.logger.info(
            "QGF_CHUNK idx=%d infer_ms=%.1f mode=%s beta=%s",
            QGFPolicyServer._qgf_chunk_counter,
            elapsed_ms,
            os.environ.get("SMOLVLA_QGF_GUIDANCE_MODE", "critic") or "critic",
            os.environ.get("SMOLVLA_QGF_BETA", "?"),
        )
        return chunk

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
        guidance_mode = os.environ.get("SMOLVLA_QGF_GUIDANCE_MODE", "critic").strip() or "critic"
        if guidance_mode not in GUIDANCE_MODES:
            raise RuntimeError(
                "SMOLVLA_QGF_GUIDANCE_MODE must be one of "
                f"{GUIDANCE_MODES}; got {guidance_mode!r}."
            )
        seed_text = os.environ.get("SMOLVLA_QGF_RANDOM_SEED", "").strip()
        random_seed = int(seed_text) if seed_text else None
        if guidance_mode == "random_matched_norm" and random_seed is None:
            raise RuntimeError(
                "SMOLVLA_QGF_RANDOM_SEED is required when "
                "SMOLVLA_QGF_GUIDANCE_MODE=random_matched_norm."
            )
        critic, metadata = load_action_chunk_critic(critic_path, device=self.device)
        critic_arch = metadata.get("critic_arch")
        if critic_arch not in {"visual_transformer", "state_action_transformer", "visual_action_transformer"}:
            raise RuntimeError(
                "The real-robot QGF server requires visual_transformer, visual_action_transformer, or "
                f"state_action_transformer critic, not {critic_arch!r}."
            )
        if int(metadata["critic_config"]["action_dim"]) != 8:
            raise RuntimeError("The deployed Armstrong critic must use eight action channels.")
        if critic_arch == "visual_transformer":
            adapter = SmolVLAVisualCriticAdapter(critic)
        elif critic_arch == "visual_action_transformer":
            adapter = SmolVLAVisualActionCriticAdapter(critic)
        else:
            adapter = SmolVLAStateActionCriticAdapter(critic)
        install_smolvla_qgf(
            self.policy,
            critic=adapter,
            config=QGuidanceConfig(
                beta=beta,
                grad_clip_norm=grad_clip_norm,
                uncertainty_scale=0.0,
                min_gate=0.0,
                guidance_mode=guidance_mode,
                random_seed=random_seed,
            ),
            critic_action_dim=8,
        )
        self.logger.info(
            "QGF single-critic guidance installed: "
            f"architecture={critic_arch}; "
            f"checkpoint={critic_path}; beta={beta:.8g}; coefficient=1/beta={1.0 / beta:.8g}; "
            f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled; "
            f"guidance_mode={guidance_mode}; random_seed={random_seed}"
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
