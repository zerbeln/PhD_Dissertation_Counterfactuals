from libc cimport math as cmath

cdef class RoverActionsSimulator:
    
    cpdef void simulate_rover_actions(
            self, 
            State state, 
            const double[:, :] rover_actions):
        cdef Py_ssize_t rover_id, n_rovers
        cdef double dx, dy, norm, clipped_action_x, clipped_action_y
        cdef double[2] rover_position_c_arr
        cdef double[:] rover_position = rover_position_c_arr
        cdef double[2] rover_orientation_c_arr
        cdef double[:] rover_orientation = rover_orientation_c_arr
        cdef const double[:, :] rover_orienations
        cdef const double[:, :] rover_positions
        
        n_rovers = state.n_rovers()
        rover_orientations = state.rover_orientations()
        rover_positions = state.rover_positions()
        
        # Translate and Reorient all rovers based on their actions
        for rover_id in range(n_rovers):
            
            # clip actions
            clipped_action_x = min(max(-1, rover_actions[rover_id, 0]), 1)
            clipped_action_y = min(max(-1, rover_actions[rover_id, 1]), 1)
    
            # turn action into global frame motion
            dx = (rover_orientations[rover_id, 0]
                * clipped_action_x
                - rover_orientations[rover_id, 1] 
                * clipped_action_y)
            dy = (rover_orientations[rover_id, 0] 
                * clipped_action_y
                + rover_orientations[rover_id, 1] 
                * clipped_action_x)
            
            # globally move and reorient agent
            rover_position[0] = rover_positions[rover_id, 0] + dx
            rover_position[1] = rover_positions[rover_id, 1] + dy
            state.set_rover_position(rover_id, rover_position)
            
            # Reorient agent in the direction of movement in
            # the global frame.  Avoid divide by 0
            # (by skipping the reorientation step entirely).
            if not (dx == 0. and dy == 0.): 
                norm = cmath.sqrt(dx*dx +  dy*dy)
                rover_orientation[0] = dx / norm
                rover_orientation[1] = dy / norm
                state.set_rover_orientation(rover_id, rover_orientation)
        