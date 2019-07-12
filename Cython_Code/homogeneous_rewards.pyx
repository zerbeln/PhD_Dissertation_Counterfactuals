"""
CODE IS OBSOLETE, DO NOT USE
"""

# import numpy as np
# from parameters import Parameters as p
# import math
# from supervisor import two_poi_case_study, four_corners_case_study
#
#
# # GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
# cpdef calc_global(rover_path, poiValueCol, poi_positions):
#     cdef int number_agents = int(p.num_rovers*p.num_types)
#     cdef int number_pois = int(p.num_pois)
#     cdef double minDistanceSqr = p.min_distance ** 2
#     cdef int historyStepCount = p.num_steps + 1
#     cdef int coupling = p.coupling
#     cdef double observationRadiusSqr = p.activation_dist ** 2
#     cdef double[:, :, :] agentPositionHistory = rover_path
#     cdef double[:, :] poiPositionCol = poi_positions
#
#     cdef int poiIndex, stepIndex, agentIndex, observerCount
#     cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
#     cdef double Inf = float("inf")
#
#     cdef double globalReward = 0.0
#
#     for poiIndex in range(number_pois):
#         closestObsDistanceSqr = Inf
#         for stepIndex in range(historyStepCount):
#             # Count how many agents observe poi, update closest distance if necessary
#             observerCount = 0
#             stepClosestObsDistanceSqr = Inf
#             for agentIndex in range(number_agents):
#                 # Calculate separation distance between poi and agent
#                 separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
#                 separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
#                 distanceSqr = separation0 * separation0 + separation1 * separation1
#
#                 # Check if agent observes poi, update closest step distance
#                 if distanceSqr < observationRadiusSqr:
#                     observerCount += 1
#                     if distanceSqr < stepClosestObsDistanceSqr:
#                         stepClosestObsDistanceSqr = distanceSqr
#
#
#             # update closest distance only if poi is observed
#             if observerCount >= coupling:
#                 if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                     closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#         # add to global reward if poi is observed
#         if closestObsDistanceSqr < observationRadiusSqr:
#             if closestObsDistanceSqr < minDistanceSqr:
#                 closestObsDistanceSqr = minDistanceSqr
#             globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#
#     return globalReward
#
#
# # DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
# cpdef calc_difference(rover_path, poiValueCol, poi_positions):
#     cdef int number_agents = p.num_rovers
#     cdef int number_pois = p.num_pois
#     cdef double minDistanceSqr = p.min_distance ** 2
#     cdef int historyStepCount = p.num_steps + 1
#     cdef int coupling = p.coupling
#     cdef double observationRadiusSqr = p.activation_dist ** 2
#     cdef double[:, :, :] agentPositionHistory = rover_path
#     cdef double[:, :] poiPositionCol = poi_positions
#
#     cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex
#     cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
#     cdef double Inf = float("inf")
#
#     cdef double globalReward = 0.0
#     cdef double globalWithoutReward = 0.0
#
#     npDifferenceRewardCol = np.zeros(number_agents)
#     cdef double[:] differenceRewardCol = npDifferenceRewardCol
#
#
#     for poiIndex in range(number_pois):
#         closestObsDistanceSqr = Inf
#         for stepIndex in range(historyStepCount):
#             # Count how many agents observe poi, update closest distance if necessary
#             observerCount = 0
#             stepClosestObsDistanceSqr = Inf
#             for agentIndex in range(number_agents):
#                 # Calculate separation distance between poi and agent
#                 separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
#                 separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
#                 distanceSqr = separation0 * separation0 + separation1 * separation1
#
#                 # Check if agent observes poi, update closest step distance
#                 if distanceSqr < observationRadiusSqr:
#                     observerCount += 1
#                     if distanceSqr < stepClosestObsDistanceSqr:
#                         stepClosestObsDistanceSqr = distanceSqr
#
#
#             # update closest distance only if poi is observed
#             if observerCount >= coupling:
#                 if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                     closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#         # add to global reward if poi is observed
#         if closestObsDistanceSqr < observationRadiusSqr:
#             if closestObsDistanceSqr < minDistanceSqr:
#                 closestObsDistanceSqr = minDistanceSqr
#             globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#
#
#     for agentIndex in range(number_agents):
#         globalWithoutReward = 0
#         for poiIndex in range(number_pois):
#             closestObsDistanceSqr = Inf
#             for stepIndex in range(historyStepCount):
#                 # Count how many agents observe poi, update closest distance if necessary
#                 observerCount = 0
#                 stepClosestObsDistanceSqr = Inf
#                 for otherAgentIndex in range(number_agents):
#                     if agentIndex != otherAgentIndex:
#                         # Calculate separation distance between poi and agent\
#                         separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
#                         separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
#                         distanceSqr = separation0 * separation0 + separation1 * separation1
#
#                         # Check if agent observes poi, update closest step distance
#                         if distanceSqr < observationRadiusSqr:
#                             observerCount += 1
#                             if distanceSqr < stepClosestObsDistanceSqr:
#                                 stepClosestObsDistanceSqr = distanceSqr
#
#
#                 # update closest distance only if poi is observed
#                 if observerCount >= coupling:
#                     if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                         closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#             # add to global reward if poi is observed
#             if closestObsDistanceSqr < observationRadiusSqr:
#                 if closestObsDistanceSqr < minDistanceSqr:
#                     closestObsDistanceSqr = minDistanceSqr
#                 globalWithoutReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#         differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
#
#     return differenceRewardCol
#
#
# # D++ REWARD ----------------------------------------------------------------------------------------------------------
# cpdef calc_dpp(rover_path, poiValueCol, poi_positions):
#     cdef int number_agents = p.num_rovers
#     cdef int number_pois = p.num_pois
#     cdef double minDistanceSqr = p.min_distance ** 2
#     cdef int historyStepCount = p.num_steps + 1
#     cdef int coupling = p.coupling
#     cdef double observationRadiusSqr = p.activation_dist ** 2
#     cdef double[:, :, :] agentPositionHistory = rover_path
#     cdef double[:, :] poiPositionCol = poi_positions
#
#     cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex, counterfactualCount
#     cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
#     cdef double Inf = float("inf")
#
#     cdef double globalReward = 0.0
#     cdef double globalWithoutReward = 0.0
#     cdef double globalWithExtraReward = 0.0
#
#     npDifferenceRewardCol = np.zeros(number_agents)
#     cdef double[:] differenceRewardCol = npDifferenceRewardCol
#
#     # Calculate Global Reward
#     for poiIndex in range(number_pois):
#         closestObsDistanceSqr = Inf
#         for stepIndex in range(historyStepCount):
#             # Count how many agents observe poi, update closest distance if necessary
#             observerCount = 0
#             stepClosestObsDistanceSqr = Inf
#             for agentIndex in range(number_agents):
#                 # Calculate separation distance between poi and agent
#                 separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
#                 separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
#                 distanceSqr = separation0 * separation0 + separation1 * separation1
#
#                 # Check if agent observes poi, update closest step distance
#                 if distanceSqr < observationRadiusSqr:
#                     observerCount += 1
#                     if distanceSqr < stepClosestObsDistanceSqr:
#                         stepClosestObsDistanceSqr = distanceSqr
#
#
#             # update closest distance only if poi is observed
#             if observerCount >= coupling:
#                 if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                     closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#         # add to global reward if poi is observed
#         if closestObsDistanceSqr < observationRadiusSqr:
#             if closestObsDistanceSqr < minDistanceSqr:
#                 closestObsDistanceSqr = minDistanceSqr
#             globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#
#     # Calculate Difference Reward
#     for agentIndex in range(number_agents):
#         globalWithoutReward = 0
#         for poiIndex in range(number_pois):
#             closestObsDistanceSqr = Inf
#             for stepIndex in range(historyStepCount):
#                 # Count how many agents observe poi, update closest distance if necessary
#                 observerCount = 0
#                 stepClosestObsDistanceSqr = Inf
#                 for otherAgentIndex in range(number_agents):
#                     if agentIndex != otherAgentIndex:
#                         # Calculate separation distance between poi and agent\
#                         separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
#                         separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
#                         distanceSqr = separation0 * separation0 + separation1 * separation1
#
#                         # Check if agent observes poi, update closest step distance
#                         if distanceSqr < observationRadiusSqr:
#                             observerCount += 1
#                             if distanceSqr < stepClosestObsDistanceSqr:
#                                 stepClosestObsDistanceSqr = distanceSqr
#
#
#                 # update closest distance only if poi is observed
#                 if observerCount >= coupling:
#                     if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                         closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#             # add to global reward if poi is observed
#             if closestObsDistanceSqr < observationRadiusSqr:
#                 if closestObsDistanceSqr < minDistanceSqr:
#                     closestObsDistanceSqr = minDistanceSqr
#                 globalWithoutReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#         differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
#
#     # Calculate Dpp Reward
#     for counterfactualCount in range(coupling):
#         # Calculate Difference with Extra Me Reward
#         for agentIndex in range(number_agents):
#             globalWithExtraReward = 0
#             for poiIndex in range(number_pois):
#                 closestObsDistanceSqr = Inf
#                 for stepIndex in range(historyStepCount):
#                     # Count how many agents observe poi, update closest distance if necessary
#                     observerCount = 0
#                     stepClosestObsDistanceSqr = Inf
#                     for otherAgentIndex in range(number_agents):
#                         # Calculate separation distance between poi and agent\
#                         separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
#                         separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
#                         distanceSqr = separation0 * separation0 + separation1 * separation1
#
#
#                         if distanceSqr < observationRadiusSqr:
#                             # Check if agent observes poi, update closest step distance
#                             observerCount += 1 + ((agentIndex == otherAgentIndex) * counterfactualCount)
#                             if distanceSqr < stepClosestObsDistanceSqr:
#                                 stepClosestObsDistanceSqr = distanceSqr
#
#                     # update closest distance only if poi is observed
#                     if observerCount >= coupling:
#                         if stepClosestObsDistanceSqr < closestObsDistanceSqr:
#                             closestObsDistanceSqr = stepClosestObsDistanceSqr
#
#                 # add to global reward if poi is observed
#                 if closestObsDistanceSqr < observationRadiusSqr:
#                     if closestObsDistanceSqr < minDistanceSqr:
#                         closestObsDistanceSqr = minDistanceSqr
#                     globalWithExtraReward += poiValueCol[poiIndex] / closestObsDistanceSqr
#             differenceRewardCol[agentIndex] = max(differenceRewardCol[agentIndex],
#             (globalWithExtraReward - globalReward)/(1.0 + counterfactualCount))
#
#
#     return differenceRewardCol
#
# # cpdef calc_sdpp(rover_path, poi_values, poi_positions):
# #     cdef int nrovers = int(p.num_rovers*p.num_types)
# #     cdef int npois = int(p.num_pois)
# #     cdef int coupling = int(p.coupling)
# #     cdef int poi_id, step_number, rover_id, rv, observer_count, od_index, other_rover_id, c
# #     cdef double min_dist = p.min_distance
# #     cdef double act_dist = p.activation_dist
# #     cdef double rover_x_dist, rover_y_dist, distance, summed_distances, current_poi_reward, temp_reward, g_without_self
# #     cdef double self_x, self_y, self_dist
# #     cdef int num_steps = int(p.num_steps + 1)
# #     cdef double inf = 1000.00
# #     cdef double g_reward = 0.0
# #     cdef double[:] difference_rewards = np.zeros(nrovers)
# #     cdef double[:] dplusplus_reward = np.zeros(nrovers)
# #
# #     # CALCULATE GLOBAL REWARD
# #     g_reward = calc_global(rover_path, poi_values, poi_positions)
# #
# #     # CALCULATE DIFFERENCE REWARD
# #     dplusplus_reward = calc_difference(rover_path, poi_values, poi_positions)
# #
# #     # CALCULATE DPP REWARD
# #     for c_count in range(coupling):
# #
# #         # Calculate Difference with Extra Me Reward
# #         for rover_id in range(nrovers):
# #             g_with_counterfactuals = 0.0
# #
# #             for poi_id in range(npois):
# #                 current_poi_reward = 0.0
# #
# #                 for step_number in range(num_steps):
# #                     observer_count = 0  # Track number of POI observers at time step
# #                     observer_distances = []
# #                     summed_distances = 0.0 # Denominator of reward function
# #                     self_x = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
# #                     self_y = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
# #                     self_dist = math.sqrt((self_x**2) + (self_y**2))
# #
# #                     # Calculate distance between poi and agent
# #                     for other_rover_id in range(nrovers):
# #                         rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
# #                         rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
# #                         distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))
# #
# #                         if distance <= min_dist:
# #                             distance = min_dist
# #                         observer_distances.append(distance)
# #
# #                         # Update observer count
# #                         if distance <= act_dist:
# #                             observer_count += 1
# #
# #                     if self_dist <= act_dist:  # Add Counterfactual Suggestions
# #                         for c in range(c_count):
# #                             if npois == 2:
# #                                 observer_distances.append(two_poi_case_study(rover_id, poi_id, self_dist))
# #                             if npois == 4:
# #                                 observer_distances.append(four_corners_case_study(rover_id, poi_id, self_dist))
# #
# #                         observer_count += c_count
# #
# #                     if observer_count >= coupling:  # If coupling satisfied, compute reward
# #                         for rv in range(coupling):
# #                             summed_distances += min(observer_distances)
# #                             od_index = observer_distances.index(min(observer_distances))
# #                             observer_distances[od_index] = inf
# #                         if summed_distances == 0:
# #                             summed_distances = -1
# #                         temp_reward = poi_values[poi_id]/summed_distances
# #                     else:
# #                         temp_reward = 0.0
# #
# #                     if temp_reward > current_poi_reward:
# #                         current_poi_reward = temp_reward
# #
# #                 g_with_counterfactuals += current_poi_reward
# #
# #             temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
# #             if temp_dpp_reward > dplusplus_reward[rover_id]:
# #                 dplusplus_reward[rover_id] = temp_dpp_reward
# #
# #     return dplusplus_reward
