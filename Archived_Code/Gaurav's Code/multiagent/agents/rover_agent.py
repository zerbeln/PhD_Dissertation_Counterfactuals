from multiagent.core import Agent
from multiagent import rendering


class RoverAgent(Agent):
    def __init__(self):
        super(RoverAgent, self).__init__()
        self.active = True
        self.type = 0

    def render(self):

        # debug
        geom = rendering.make_circle(self.size * 2.0)
        geom.set_color(*self.color, alpha=0.2)
        xform = rendering.Transform()
        xform.set_translation(*self.state.p_pos)
        geom.add_attr(xform)

        # visual quadrants
        axis_length = 2
        x_axis = rendering.Line(
            start=(
                self.state.p_pos[0] - axis_length,
                self.state.p_pos[1]
            ),
            end=(
                self.state.p_pos[0] + axis_length,
                self.state.p_pos[1]
            )
        )
        x_axis.set_color(*self.color, alpha=0.3)

        y_axis = rendering.Line(
            start=(
                self.state.p_pos[0],
                self.state.p_pos[1] - axis_length
            ),
            end=(
                self.state.p_pos[0],
                self.state.p_pos[1] + axis_length
            )
        )
        y_axis.set_color(*self.color, alpha=0.3)

        return [geom, x_axis, y_axis]
