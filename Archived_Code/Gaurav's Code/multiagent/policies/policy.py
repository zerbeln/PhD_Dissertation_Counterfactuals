# individual agent policy
class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()

    def transition(self,
                   observation,
                   action,
                   reward,
                   next_observation,
                   done):
        pass
