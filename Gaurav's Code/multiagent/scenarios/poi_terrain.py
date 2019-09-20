import multiagent.utilities.logging as logging
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.agents.rover_agent import RoverAgent
from multiagent.agents.rover_poi import RoverLandmark
from multiagent.scenario import BaseScenario
from multiagent.utilities.rover_specs import RoverSpec, RewardType
import math
import random

logger = logging.getLogger(__name__)


class Scenario(BaseScenario):

    def __init__(self):
        self._cached_gloabl = 0
        self._cached_gloabl_valid = False
        self.n_to_observe = 3
        self.env_spec = None

    def make_world(self, roverSpec: RoverSpec):

        world = World()
        self.env_spec = roverSpec
        num_poi = roverSpec.poi_count
        num_agents = roverSpec.agent_count
        reward_type = roverSpec.reward_type

        # add agents
        world.agents = [RoverAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [RoverLandmark() for i in range(num_poi)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'poi %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self._reset_world(world, True)
        self.reward_type = reward_type
        # self._obs_normalize(world)
        return world

    def _reset_world(self, world, copy_spec=True):
        for i, agent in enumerate(world.agents):
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.type = self.env_spec.agent_specs[i].type
            if copy_spec:
                if(agent.state.p_pos is None):
                    agent.state.p_pos = np.empty_like(
                        self.env_spec.agent_specs[i].p_pos)
                agent.state.p_pos[:] = self.env_spec.agent_specs[i].p_pos
                if(agent.color is None):
                    agent.color = np.empty_like(
                        self.env_spec.agent_specs[i].color)
                agent.color[:] = self.env_spec.agent_specs[i].color
            else:
                spawn_range = self.env_spec.spawn_range
                agent.state.p_pos[0] = np.random.uniform(
                    -spawn_range[0], spawn_range[0])
                agent.state.p_pos[1] = np.random.uniform(
                    -spawn_range[1], spawn_range[1])
                agent.color = np.array(
                    [np.random.uniform(low=0.0, high=1),
                     np.random.uniform(low=0.0, high=1),
                     np.random.uniform(low=0.0, high=1)])

        for i, landmark in enumerate(world.landmarks):
            landmark.color = self.env_spec.poi_specs[i].color
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.c = np.zeros(world.dim_c)
            landmark.density = self.env_spec.poi_specs[i].value  # poi value
            landmark.observation_radius = self.env_spec.poi_specs[i].observation_radius

            if(landmark.state.p_pos is None):
                landmark.state.p_pos = np.empty_like(
                    self.env_spec.poi_specs[i].p_pos)
            landmark.state.p_pos[:] = self.env_spec.poi_specs[i].p_pos
            # NOTE Enable shuffling POIs by un-commenting the comment block below
            """
            if copy_spec:
                if(landmark.state.p_pos is None):
                    landmark.state.p_pos = np.empty_like(
                        self.env_spec.poi_specs[i].p_pos)
                landmark.state.p_pos[:] = self.env_spec.poi_specs[i].p_pos
            else:
                spawn_range = self.env_spec.spawn_range
                landmark.state.p_pos[0] = np.random.uniform(
                    -spawn_range[0], spawn_range[0])
                landmark.state.p_pos[1] = np.random.uniform(
                    -spawn_range[1], spawn_range[1])
            """

    def reset_world(self, world):

        return self._reset_world(world, not self.env_spec.shuffle_on_reset)
        # world.landmarks[0].state.p_pos = np.array([0., 0.])
        # world.landmarks[0].color = np.array([0., 0., 0.])

        # world.agents[0].color = np.array([0., 0., 0.])
        # world.agents[1].color = np.array([0., 1., 0.])
        # world.agents[2].color = np.array([0., 0., 1.])

    def distance(self, entityA, entityB):
        y2 = entityB.state.p_pos[1]
        y1 = entityA.state.p_pos[1]
        x2 = entityB.state.p_pos[0]
        x1 = entityA.state.p_pos[0]
        distance = math.sqrt(
            math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
        distance = max(distance, self.env_spec.minimum_distance_to_poi)
        return distance

    def vec_distance(self, entityA, entityB):
        distance = np.sum(np.square(entityA.state.p_pos - entityB.state.p_pos))
        distance = max(distance, self.env_spec.minimum_distance_to_poi)
        return distance

    def reward(self, agent, world):
        if(self.reward_type == RewardType.Local):
            return self.local_reward(agent, world)

        if(self.reward_type == RewardType.Global):
            # return self.single_agent_gloabl_reward(agent, world)
            return self.global_reward(agent, world)

        if(self.reward_type == RewardType.Difference):
            return self.difference_reward(agent, world)

        if(self.reward_type == RewardType.Dpp):
            return self.dpp_reward(agent, world)

        if(self.reward_type == RewardType.HDpp):
            return self.dpp_reward(agent, world)

    def local_reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos -
                                 world.landmarks[0].state.p_pos))
        return -dist2

    def invalidate_cache(self):
        self._cached_gloabl_valid = False

    def global_reward(self, agent, world):

        # if has a valid cache, return it
        if(self._cached_gloabl_valid is True):
            return self._cached_gloabl

        # cache was not valid, will calculate global now and set it to valid
        self._cached_gloabl = self._global_reward(agent, world)
        self._cached_gloabl_valid = True
        return self._cached_gloabl

    def _global_reward(self, agent, world, counterfactuals=[]):

        reward = 0
        for poi in world.landmarks:
            distances = []
            candidates = []
            for ag in world.agents:
                if(ag.active is False):
                    # pretend this agent does not exist in the environment
                    continue

                distance = self.vec_distance(poi, ag)
                if (distance <= poi.observation_radius):
                    distances.append(distance)
                    candidates.append(ag)

            for ag in counterfactuals:
                distance = self.vec_distance(poi, ag)
                if (distance <= poi.observation_radius):
                    distances.append(distance)
                    candidates.append(ag)

            if(len(distances) == self.n_to_observe):
                """
                reward += poi.density / \
                    max(self.env_spec.minimum_distance_to_poi, np.average(distance))
                """
                # type config
                if(any(x.type == 0 for x in candidates) and any(x.type == 1 for x in candidates) and any(x.type == 2 for x in candidates)):
                    # print ("all three types!, with poi_reward: ", poi_reward)
                    reward += poi.density / max(0.00001, np.average(distance))
        return reward

    def difference_reward(self, agent, world):

        # check if global reward provides enough signal
        g_reward = self.global_reward(agent, world)
        # if(g_reward > 0):
        #    return g_reward

        # calculate global without this agent:
        agent.active = False
        g_without_me = self._global_reward(agent, world)
        agent.active = True

        difference_reward = g_reward - g_without_me
        return difference_reward

    def _dpp_reward(self, agent, world, n):

        # create n counterfactuals
        counterfactuals = [RoverAgent() for i in range(n)]
        for i, cagent in enumerate(counterfactuals):
            cagent.name = 'agent [cf] %d' % i
            cagent.collide = False
            cagent.silent = True
            cagent.state.p_pos = np.copy(agent.state.p_pos)
            cagent.state.p_vel = np.copy(agent.state.p_vel)
            cagent.state.c = np.copy(agent.state.c)

            if(self.reward_type == RewardType.HDpp):
                cagent.type = random.choice([0, 1, 2])
            else:
                cagent.type = 0

        c_reward = self._global_reward(
            agent, world, counterfactuals=counterfactuals)
        dpp_r = (c_reward - self.global_reward(agent, world)) * 1.0 / n
        return dpp_r

    def dpp_reward(self, agent, world):

        d_reward = self.difference_reward(agent, world)
        # dpp_n = self._dpp_reward(agent, world, len(world.agents))

        # if(dpp_n <= d_reward):
        #     return d_reward
        if (d_reward > 0):
            return d_reward

        n = 0
        dpp_n0 = d_reward
        while(n < len(world.agents)):
            n = n + 1
            dpp_n1 = self._dpp_reward(agent, world, n)
            if (dpp_n1 > dpp_n0):
                return dpp_n1
        return d_reward

    def single_agent_gloabl_reward(self, agent, world):
        reward = 0
        for poi in world.landmarks:
            distances = []
            for ag in world.agents:
                distance = self.vec_distance(poi, ag)
                distances.append(distance)
            min_distance = min(distances)
            if(min_distance <= poi.observation_radius):
                reward += poi.density / min_distance

        return reward

    def addToQuad(self, entity, angle, quads):

        if(angle > math.pi * 0.5):
            quads[1].append(entity)
        elif(angle >= 0):
            quads[0].append(entity)
        elif(angle < -math.pi * 0.5):
            quads[2].append(entity)
        elif(angle < 0):
            quads[3].append(entity)
        else:
            logger.warning("Invalid quad!")

    def _obs_normalize(self, world):
        max_value = 0
        min_value = 0
        for w_poi in world.landmarks:
            max_value += w_poi.density / self.env_spec.minimum_distance_to_poi
            min_value += w_poi.density / (self.env_spec.width * 4)
            # print("> w_poi.density: ", w_poi.density)
            # print("> self.env_spec.width * 2.0: ", self.env_spec.width)
            # print("> min_value: ", min_value)

        # print("max_value: ", max_value)
        # print("min_value: ", min_value)

        """
        distance = self.vec_distance(world.landmarks[0], world.agents[0])
        desnsity = world.landmarks[0].density / distance
        print("world.landmarks[0].density: ", world.landmarks[0].density)
        print("Distance: ", distance)
        print("Desnsity: ", desnsity)
        print("bound distance: ", self.env_spec.width * 2)
        """
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        """
        normalized = (((desnsity - min_value) * (1 - 0)) /
                      (max_value - min_value)) + 0
        print("Normalized: ", normalized)
        """
        self.max_obs_density = max_value
        self.min_obs_density = 0  # min_value
        # print("self.max_obs_density: ", self.max_obs_density)
        # print("self.min_obs_density: ", self.min_obs_density)

    def _normalize_obs_density(self, density):
        normalized = (((density - self.min_obs_density) * (1 - 0)) /
                      (self.max_obs_density - self.min_obs_density)) + 0
        return normalized

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        agent_quads = [[] for i in range(4)]
        poi_quads = [[] for i in range(4)]

        for w_agent in world.agents:
            if w_agent == agent:
                continue
            p1 = agent.state.p_pos
            p2 = w_agent.state.p_pos
            angle = (math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            self.addToQuad(w_agent, angle, agent_quads)

        for w_poi in world.landmarks:
            p1 = agent.state.p_pos
            p2 = w_poi.state.p_pos
            angle = (math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            self.addToQuad(w_poi, angle, poi_quads)

        # print(chr(27) + "[2J")
        # print("_________________", agent.name, "_________________")
        # print ("agent_quads: ", agent_quads)
        # print("______")
        # print ("poi_quads: ", poi_quads)

        agent_input = np.empty(4)
        for i, quad in enumerate(agent_quads):
            a_value = 0
            for a in quad:
                y2 = a.state.p_pos[1]
                y1 = agent.state.p_pos[1]
                x2 = a.state.p_pos[0]
                x1 = agent.state.p_pos[0]
                distance = math.sqrt(
                    math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
                distance = max(distance, self.env_spec.minimum_distance_to_poi)
                a_value += 1 / distance
            agent_input[i] = a_value

        poi_input = np.empty(4)
        for i, quad in enumerate(poi_quads):
            p_value = 0
            for poi in quad:
                y2 = poi.state.p_pos[1]
                y1 = agent.state.p_pos[1]
                x2 = poi.state.p_pos[0]
                x1 = agent.state.p_pos[0]
                distance = math.sqrt(
                    math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
                distance = max(distance, self.env_spec.minimum_distance_to_poi)
                p_value += poi.density / distance
            # print("Original pvalue: ", p_value)
            # print("Normalized: ", self._normalize_obs_density(p_value))
            poi_input[i] = p_value

        return np.concatenate([poi_input] + [agent_input])

    def done(self, agent, world):
        # TODO Add these. Do they exist for every agent?
        return False
        bound = self.env_spec.zoom
        if agent.state.p_pos[0] > bound or agent.state.p_pos[0] < -bound or agent.state.p_pos[1] > bound or agent.state.p_pos[1] < -bound:
            return True
        else:
            return False
