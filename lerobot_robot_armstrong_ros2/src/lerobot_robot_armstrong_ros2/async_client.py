"""Armstrong-specific LeRobot async client with overlapping action chunks."""

import logging
import signal
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Queue

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedObservation, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins


class ArmstrongRobotClient(RobotClient):
    """Request the next chunk before the current policy queue becomes empty.

    Upstream marks an observation ``must_go`` only after the queue is empty.
    That is safe for high-latency action units but causes gaps for our radian
    state, because the policy server otherwise regards nearby states as
    similar.  This override marks the threshold observation as mandatory, so
    the normal upstream timestamp aggregation overlaps old and new chunks.
    """

    def _sync_policy_queue_status(self) -> tuple[int, int, bool]:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        expected = max(1, int(self.config.actions_per_chunk))
        # After an episode reset the new observation timestep equals the last
        # executed timestep. Upstream correctly drops that duplicate first
        # action, leaving 49 executable actions from a nominal 50-step chunk.
        # Treat this de-duplicated chunk as fully preloaded.
        minimum_ready = max(1, expected - 1)
        ready = self.action_chunk_size >= minimum_ready and queue_size >= minimum_ready
        self.robot.update_policy_queue_state(queue_size, expected, ready)
        return queue_size, expected, ready

    def reset_episode_queue(self) -> None:
        """Drop stale unexecuted actions without reconnecting or reloading the model."""
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_chunk_size = -1
        self.must_go.set()
        self._sync_policy_queue_status()
        self.logger.info("EPISODE_QUEUE_RESET latest_action=%s", self.latest_action)

    def actions_available(self):
        available = super().actions_available()
        self._sync_policy_queue_status()
        return available

    def _ready_to_send_observation(self):
        # Before the operator enables policy motion, retain the complete first
        # chunk instead of consuming or continuously replacing it.  This makes
        # the MOVE prompt mean exactly "50 executable actions are preloaded".
        if not self.robot.action_enabled and self.actions_available():
            return False
        return super()._ready_to_send_observation()

    def control_loop_observation(self, task: str, verbose: bool = False):
        try:
            if not self.running:
                return None
            start_time = time.perf_counter()
            # The Orin currently needs roughly 1.5-1.7 seconds to infer a
            # 50-action chunk.  Re-publish the most recent guarded joint
            # target at the configured client rate while a replacement chunk
            # is in flight.  If this process dies, the lower-level watchdog
            # still stops the arm after its configured timeout.
            self.robot.resend_last_joint_command()
            raw_observation = self.robot.get_observation()
            raw_observation["task"] = task
            with self.latest_action_lock:
                latest_action = self.latest_action
            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
                threshold_reached = (
                    self.action_chunk_size > 0
                    and current_queue_size / self.action_chunk_size
                    <= self._chunk_size_threshold
                )
                observation.must_go = self.must_go.is_set() and (
                    self.action_chunk_size <= 0 or threshold_reached
                )
            if not self.running:
                return None
            sent = self.send_observation(observation)
            if sent:
                self.robot.publish_policy_observation(
                    observation.get_timestep(), raw_observation
                )
            if sent and observation.must_go:
                self.must_go.clear()
            if verbose:
                elapsed = time.perf_counter() - start_time
                self.logger.info(
                    "Obs #%s queue=%s must_go=%s capture_send_ms=%.2f",
                    observation.get_timestep(),
                    current_queue_size,
                    observation.must_go,
                    elapsed * 1000,
                )
            return raw_observation
        except Exception as exc:
            self.logger.error("Error in Armstrong observation sender: %s", exc)
            return None

    def control_loop_action(self, verbose: bool = False):
        if not self.robot.action_enabled:
            self._sync_policy_queue_status()
            return None
        try:
            start = time.perf_counter()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
            self.robot.set_policy_action_timestep(timed_action.get_timestep())
            result = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            if verbose:
                with self.action_queue_lock:
                    current_queue_size = self.action_queue.qsize()
                self.logger.debug(
                    "Action #%s performed; queue=%s pop_send_ms=%.2f",
                    timed_action.get_timestep(),
                    current_queue_size,
                    (time.perf_counter() - start) * 1000,
                )
            self._sync_policy_queue_status()
            return result
        except Exception as exc:
            # A short feedback pause must not tear down the whole client. The
            # failed action was already removed from the queue; later actions
            # resume after feedback becomes fresh, while the last guarded
            # target continues to be refreshed by control_loop_observation.
            self.logger.error("Error in Armstrong action sender: %s", exc)
            return None

    def log_runtime_status(self) -> None:
        last_signature = None
        last_log_time = 0.0
        while self.running:
            queue_size, expected, ready = self._sync_policy_queue_status()
            with self.latest_action_lock:
                latest_action = self.latest_action
            mode = "RUNNING" if self.robot.action_enabled else "PRELOAD"
            signature = (mode, ready, queue_size // 10, latest_action // 10)
            now = time.monotonic()
            if signature != last_signature or now - last_log_time >= 5.0:
                self.logger.info(
                    "POLICY_RUNTIME mode=%s first_chunk_ready=%s queue=%s/%s latest_action=%s",
                    mode,
                    int(ready),
                    queue_size,
                    expected,
                    latest_action,
                )
                last_signature = signature
                last_log_time = now
            self.shutdown_event.wait(0.25)


@draccus.wrap()
def main(cfg: RobotClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    client = ArmstrongRobotClient(cfg)
    client.robot.set_episode_reset_callback(client.reset_episode_queue)

    def request_stop(signum, _frame) -> None:
        client.logger.warning("Stop signal %s received; stopping the async client", signum)
        client.shutdown_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not client.start():
        raise SystemExit(2)
    action_receiver = threading.Thread(target=client.receive_actions, daemon=True)
    runtime_logger = threading.Thread(target=client.log_runtime_status, daemon=True)
    action_receiver.start()
    runtime_logger.start()
    try:
        client.control_loop(task=cfg.task)
    finally:
        client.shutdown_event.set()
        action_receiver.join(timeout=5.0)
        runtime_logger.join(timeout=2.0)
        client.stop()
        action_receiver.join(timeout=2.0)
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
