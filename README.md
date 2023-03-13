# PhD Research Repository

This repository contains the code I am using for my dissertation research for my PhD in Robotics from Oregon State University. This codebase features a heavily modified version of the rover domain code contained in my RoverDomain repository that is used by the AADI lab. There are three sub-projects contained within this repository. Each project constitutes a research contribution from my dissertation work.

# Counterfactual Focused Learning (CFL)
The contribution of this research is to creation and apply more intelligent counterfactual states that can be used within counterfactual-based reward shaping approaches (such as Difference Evaluations and D++) to focus agent learning on specific niches within the behavioral space (specific areas of the state-space). This work demonstrates that agent teams achieve coordination more reliably resulting in better overall performance compared to using singular or random counterfactuals.

Skills: Cooperative CoEvolutionary Algorithms, Neural Networks, Reward Shaping, Difference Evaluations, D++, Multiagent Coordination

Publications:
- The Power of Suggestion: https://par.nsf.gov/biblio/10197807

# Counterfactual Knowledge Injection (CKI)
The contribution of this research is the introduction of perception shaping to inject additional knowledge into multiagent systems. Perception shaping re-applies the concept of using counterfactual states to shape agent information outside of the learning process. Using counterfactual states within reward functions can shape agent rewards to provide individuals with more insightful feedback on the effectiveness of their policies or actions. This work demonstrates that this concept can be applied outside of the learning paradigm to shape agent perceptions of their environment allowing agents to adapt to changes without needing to retrain or relearn.

Skills: Cooperative CoEvolutionary Algorithms, Neural Networks, Reward Shaping, Difference Evaluations, D++, Multiagent Coordination, Adaptive Agents

Publications: (Work for this contribution is currently under review for publication.)

# Autonomous Counterfactual Generation via Supervisory Agents (ACG)
The contribution of this work is to introduce a top-level, supervisory agent that uses the concept of perception shaping to provide low-level agents with augmented state information that can help them overcome changes to the environment or task they cannot perceive on their own. Supervisor agents cannot act directly upon the environment; however, they have a greater overall state-perception than low-level agents. Using this greater state-knowledge, the supervisor constructs counterfactual states to shape low-level agent perceptions providing them with representations of critical but missing state information.

Skills: Cooperative CoEvolutionary Algorithms, Neural Networks, Reward Shaping, Difference Evaluations, D++, Multiagent Coordination, Adaptive Agents

Publications: (Work for this contribution is currently under review for publication.)
