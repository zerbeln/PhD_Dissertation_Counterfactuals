from multiagent.core import Landmark
from multiagent import rendering


class RoverLandmark(Landmark):

    def __init__(self):
        super(RoverLandmark, self).__init__()
        self.observation_radius = 1.0

    def render(self):

        # oberservation radius
        geom = rendering.make_circle(self.observation_radius)
        geom.set_color(*self.color, alpha=0.05)
        xform = rendering.Transform()
        xform.set_translation(*self.state.p_pos)
        geom.add_attr(xform)

        return [geom]
