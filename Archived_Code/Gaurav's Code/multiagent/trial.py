
import multiagent.utilities.visualization as visualization
from multiagent.scenarios.poi_terrain import Scenario
from multiagent.environment import MultiAgentEnv
from multiagent.policies.q_policy import QPolicy
from multiagent.policies.pr_q_policy import PRQPolicy
from multiagent.policies.interactive_policy import InteractivePolicy

from multiagent.utilities.rover_specs import RoverSpec, RewardType, POISpec, AgentSpec

import numpy as np
import tensorflow as tf
import time
import sys

from multiagent.utilities.logging import getLogger
logger = getLogger(__name__)


if __name__ == "__main__":

    reward_type = RewardType.Global
    visualization._type_string = "Global"
    if(len(sys.argv) > 1):
        if(sys.argv[1] == 'd'):
            reward_type = RewardType.Difference
            visualization._type_string = "Difference"
            logger.info("Using Difference Reward.")
        elif(sys.argv[1] == 'p'):
            reward_type = RewardType.Dpp
            visualization._type_string = "Dpp"
            logger.info("Using DPP Reward.")
        elif(sys.argv[1] == 'x'):
            reward_type = RewardType.HDpp
            visualization._type_string = "H-Dpp"
            logger.info("Using H-DPP Reward.")
    else:
        logger.info("Using Global Reward.")

    env_spec = RoverSpec(
        width=5,
        height=5,
        poi_count=3,
        poi_specs=[POISpec(observation_radius=1, p_pos=np.zeros(2)),
                   # POISpec(observation_radius=3,
                   #         color=np.array([1.0, 0., 0.])),
                   # POISpec(value=20)
                   ],
        agent_count=9,
        # agent_specs=[AgentSpec(p_pos=np.array([0., -2.0]))],
        agent_specs=[
            AgentSpec(type=0), AgentSpec(type=1), AgentSpec(type=2),
            AgentSpec(type=0), AgentSpec(type=1), AgentSpec(type=2),
            AgentSpec(type=0), AgentSpec(type=1), AgentSpec(type=2)
        ],
        reward_type=reward_type,
        reward_type_string=visualization._type_string,
        shuffle_on_reset=True
    )

    scenario = Scenario()
    world = scenario.make_world(env_spec)
    env = MultiAgentEnv(
        world,
        env_spec,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.done,
        shared_viewer=True
    )
    # env.viewers[0].cam_range = env_spec.zoom

    """
    # interactive policy test
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    obs_n = env.reset()
    world.agents[0].state.p_pos = [0., 0.]
    while True:
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        obs_n, reward_n, done_n, _ = env.step(act_n)
        env.render()
    """
    output_visualization = True

    policies = [
        QPolicy(
            env,
            i,
            5,
            8,
            replace_target_iter=500,
            learning_rate=0.001,
            epsilon=0.99,
            memory_size=100000,
            output_visualization=output_visualization
        )
        for i in range(env.n)
    ]

    """
    while True:
        env.reset()
        time.sleep(1)
        env.render()
    """
    if(output_visualization):
        summary_writer = tf.summary.FileWriter(
            visualization.train_path()+"/reward")

    env.viewers[0].window.set_caption(visualization._type_string)

    episodes = 100000
    steps_per_episode = 150
    render_after_episodes = 20000

    for episode in range(episodes):
        observations = env.reset()
        """
        print("Observations: ", observations)
        while True:
            env.render()
        exit(1)
        """
        steps = 0
        episode_reward = 0
        while True:

            scenario.invalidate_cache()
            steps += 1
            if(episode > render_after_episodes):
                env.render()
            actions = []
            for i, policy in enumerate(policies):
                actions.append(policy.action(observations[i]))

            next_observations, rewards, dones, _ = env.step(actions)

            for i, policy in enumerate(policies):
                policy.transition(
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    dones[i]
                )

            observations = next_observations
            episode_reward += np.sum(rewards)

            # if any ogent is done, end episode
            if np.any(dones) or steps >= steps_per_episode:
                break

        if(output_visualization):
            summary = tf.Summary()
            summary.value.add(tag='Episode_reward',
                              simple_value=episode_reward)
            summary_writer.add_summary(summary, episode)
            summary_writer.flush()
        logger.debug("Completed episode %s in %s steps, with total reward %s. [epsilon %s]",
                     episode, steps, episode_reward, policies[0].brain.epsilon)

    logger.debug("Trial complete.")
