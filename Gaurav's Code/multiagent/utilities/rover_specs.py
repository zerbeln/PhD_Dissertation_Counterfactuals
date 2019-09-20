from enum import Enum
import numpy as np


class EnvSpec:
    def __init__(self):
        pass


RewardType = Enum('RewardType', 'Local Global Difference Dpp HDpp')


class RoverSpec(EnvSpec):
    def __init__(self,
                 width=1,
                 height=1,
                 poi_count=1,
                 poi_specs=[],
                 agent_count=1,
                 agent_specs=[],
                 reward_type=0,
                 reward_type_string="global reward",
                 shuffle_on_reset=False
                 ):
        super(RoverSpec, self).__init__()
        self.width = width
        self.height = height
        self.poi_count = poi_count
        self.poi_specs = poi_specs
        self.agent_count = agent_count
        self.agent_specs = agent_specs
        self.reward_type = reward_type
        self.reward_type_string = reward_type_string
        self.shuffle_on_reset = shuffle_on_reset
        self.minimum_distance_to_poi = 0.1
        self.zoom = 1
        self.spawn_range = [self.width, self.height]
        self._compute_zoom()

        if(len(poi_specs) < poi_count):
            index = len(poi_specs)
            for i in range(index, poi_count):
                self.poi_specs.append(POISpec())

        if(len(agent_specs) < agent_count):
            index = len(agent_specs)
            for i in range(index, agent_count):
                self.agent_specs.append(AgentSpec())

        for pspec in self.poi_specs:
            pspec._default_pos(self.spawn_range)

        for aspec in self.agent_specs:
            aspec._default_pos(self.spawn_range)

    def _compute_zoom(self):
        dimension = self.width if self.width > self.height else self.height
        self.zoom = dimension + 0.2  # slightly bigger view.


class POISpec:
    def __init__(self,
                 observation_radius=1,
                 value=0.5,
                 p_pos=None,
                 color=None
                 ):
        self.observation_radius = observation_radius
        self.value = value
        self.p_pos = p_pos
        self.color = color if color is not None else np.zeros(3)

    def _default_pos(self, spawn_range):
        if(self.p_pos is None):
            self.p_pos = np.zeros(2)
            self.p_pos[0] = np.random.uniform(-spawn_range[0], spawn_range[0])
            self.p_pos[1] = np.random.uniform(-spawn_range[1], spawn_range[1])


class AgentSpec:
    def __init__(self,
                 p_pos=None,
                 color=None,
                 type=0
                 ):
        self.p_pos = p_pos
        self.color = color if color is not None else np.array(
            [np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.0, high=1)])
        self.type = type

    def _default_pos(self, spawn_range):
        if(self.p_pos is None):
            self.p_pos = np.zeros(2)
            self.p_pos[0] = np.random.uniform(-spawn_range[0], spawn_range[0])
            self.p_pos[1] = np.random.uniform(-spawn_range[1], spawn_range[1])
