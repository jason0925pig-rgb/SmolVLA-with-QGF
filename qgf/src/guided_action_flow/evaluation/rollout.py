from __future__ import annotations

from guided_action_flow.evaluation.metrics import EpisodeMetrics
from guided_action_flow.types import PolicyInput, Transition


def run_episode(env, policy, reward_fn, episode_index: int, seed: int | None = None):
    obs = env.reset(seed=seed)
    total_reward = 0.0
    success = False
    steps = 0

    while True:
        policy_output = policy.act_chunk(
            PolicyInput(
                observation=obs,
                instruction=env.task_instruction,
                proprio=obs.get("proprio"),
            )
        )
        action_chunk = policy_output.action_chunk
        for action in action_chunk:
            next_obs, env_reward, done, info = env.step(action)
            transition = Transition(
                observation=obs,
                action=action,
                reward=env_reward,
                next_observation=next_obs,
                done=done,
                info=info,
            )
            reward_output = reward_fn(transition)
            total_reward += reward_output.reward
            success = success or bool(reward_output.diagnostics.get("success", 0.0))
            steps += 1
            obs = next_obs
            if done:
                return EpisodeMetrics(episode_index, success, total_reward, steps)

