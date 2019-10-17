from libc cimport math as cmath
import numpy as np
    
cdef class RoverObservationsGetter:
    
    def __init__(self):
        self.m_rover_observations = None
        self.m_n_rover_observation_sections = 1
        self.m_min_dist = 1.
        self.m_n_rovers = 0
        
    
    cpdef const double[:, :] rover_observations(self, State state):
        cdef Py_ssize_t rover_id, poi_id, other_rover_id, sec_id, obs_id
        cdef Py_ssize_t n_rovers 
        cdef double gf_displ_x, gf_displ_y
        cdef double rf_displ_x, rf_displ_y, dist, angle
        cdef const double[:, :] rover_position
        cdef const double[:, :] rover_orientations
        cdef const double[:, :] poi_positions
        cdef const double[:] poi_values
        
        n_rovers = state.n_rovers()
        n_pois = state.n_pois()
        rover_positions = state.rover_positions()
        rover_orientations = state.rover_orientations()
        poi_positions = state.poi_positions()
        poi_values = state.poi_values()
        
        # Zero all observations.
        self.m_rover_observations[...] = 0.
        
        # Calculate observation for each rover.
        for rover_id in range(n_rovers):
            
            # Update rover type observations
            for other_rover_id in range(n_rovers):
                # Agents should not sense self, ergo skip self comparison.
                if rover_id == other_rover_id:
                    continue
                    
                # Get global (gf) frame displacement.
                gf_displ_x = (
                    rover_positions[other_rover_id, 0]
                    - rover_positions[rover_id, 0])
                gf_displ_y = (
                    rover_positions[other_rover_id, 1] 
                    - rover_positions[rover_id, 1])
                    
                # Get rover frame (rf) displacement.
                rf_displ_x = (
                    rover_orientations[rover_id, 0] 
                    * gf_displ_x
                    + rover_orientations[rover_id, 1]
                    * gf_displ_y)
                rf_displ_y = (
                    rover_orientations[rover_id, 0]
                    * gf_displ_y
                    - rover_orientations[rover_id, 1]
                    * gf_displ_x)
                    
                dist = cmath.sqrt(rf_displ_x*rf_displ_x + rf_displ_y*rf_displ_y)
                
                # By bounding distance value we 
                # implicitly bound sensor values (1/dist^2) so that they 
                # don't explode when dist = 0.
                if dist < self.m_min_dist:
                    dist = self.m_min_dist
                    
                # Get arc tangent (angle) of displacement.
                angle = cmath.atan2(rf_displ_y, rf_displ_x) 
                
                #  Get intermediate Section Index by discretizing angle.
                sec_id = <Py_ssize_t>cmath.floor(
                    (angle + cmath.pi)
                    / (2 * cmath.pi) 
                    * self.m_n_rover_observation_sections)
                    
                # Clip section index for pointer safety.
                obs_id = (
                    min(
                        max(0, sec_id), 
                        self.m_n_rover_observation_sections - 1))
                    
                self.m_rover_observations[rover_id, obs_id] += 1. / (dist*dist)

            # Update POI type observations.
            for poi_id in range(n_pois):
            
                # Get global (gf) frame displacement.
                gf_displ_x = (
                    poi_positions[poi_id, 0]
                    - rover_positions[rover_id, 0])
                gf_displ_y = (
                    poi_positions[poi_id, 1] 
                    - rover_positions[rover_id, 1])
                    
                # Get rover frame (rf) displacement.
                rf_displ_x = (
                    rover_orientations[rover_id, 0] 
                    * gf_displ_x
                    + rover_orientations[rover_id, 1]
                    * gf_displ_y)
                rf_displ_y = (
                    rover_orientations[rover_id, 0]
                    * gf_displ_y
                    - rover_orientations[rover_id, 1]
                    * gf_displ_x)
                    
                dist = cmath.sqrt(rf_displ_x*rf_displ_x + rf_displ_y*rf_displ_y)
                
                # By bounding distance value we 
                # implicitly bound sensor values (1/dist^2) so that they 
                # don't explode when dist = 0.
                if dist < self.m_min_dist:
                    dist = self.m_min_dist
                    
                # Get arc tangent (angle) of displacement.
                angle = cmath.atan2(rf_displ_y, rf_displ_x) 
                
                #  Get intermediate Section Index by discretizing angle.
                sec_id = <Py_ssize_t>cmath.floor(
                    (angle + cmath.pi)
                    / (2 * cmath.pi) 
                    * self.m_n_rover_observation_sections)
                    
                # Clip section index for pointer safety and offset observations
                # index for POIs.
                obs_id = (
                    min(
                        max(0, sec_id), 
                        self.m_n_rover_observation_sections - 1)
                    + self.m_n_rover_observation_sections)
                    
                self.m_rover_observations[rover_id, obs_id] += (
                    poi_values[poi_id] / (dist*dist))
        
        return self.m_rover_observations
        
    cpdef Py_ssize_t n_rovers(self):
        return self.m_n_rovers
        
    cpdef double min_dist(self):
        return self.m_min_dist
        
    cpdef Py_ssize_t n_rover_observation_sections(self):
        return self.m_n_rover_observation_sections
    
    cpdef void set_n_rovers(self, Py_ssize_t n_rovers):
        if n_rovers < 0:
            raise ValueError(
                "The number of rovers (n_rovers) must be non-negative. "
                + "The number of rovers received is %d."
                % n_rovers)
                
        self.m_n_rovers = n_rovers
        
        self.m_rover_observations = (
            np.zeros(
                (self.m_n_rovers, 2 * self.m_n_rover_observation_sections)))
        
        
    cpdef void set_min_dist(self, double min_dist):
        if min_dist <= 0:
            raise ValueError("Minimum distance (min_dist) must be positive. "
                + "A value of %d was received"
                % min_dist)
                
        self.m_min_dist = min_dist
        
        
    cpdef void set_n_rover_observation_sections(
            self, 
            Py_ssize_t n_rover_observation_sections):
        if n_rover_observation_sections <= 0:
            raise ValueError("Number of rover_observation sections "
                + "(n_rover_observations_sections) must be "
                + "positive. A value of %d was received"
                % n_rover_observation_sections)
                
        self.m_n_rover_observation_sections = n_rover_observation_sections
        
        self.m_rover_observations = (
            np.zeros(
                (self.m_n_rovers, 2 * self.m_n_rover_observation_sections)))

        