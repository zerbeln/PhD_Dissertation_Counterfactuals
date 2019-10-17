
import numpy as np

cdef class State:
    def __init__(self):
        self.m_n_rovers = 0
        self.m_n_pois = 0
        self.m_init_rover_positions = None
        self.m_init_rover_orientations = None
        self.m_rover_positions = None
        self.m_rover_orientations = None
        self.m_poi_values = None
        self.m_poi_positions = None
        
    cpdef void reset(self):
        self.m_rover_positions[...] = self.m_init_rover_positions
        self.m_rover_orientations[...] = self.m_rover_orientations
    
    cpdef Py_ssize_t n_rovers(self):
        return self.m_n_rovers
        
    cpdef Py_ssize_t n_pois(self):
        return self.m_n_pois
        
    cpdef const double[:, :] init_rover_positions(self):
        return self.m_init_rover_positions
        
    cpdef const double[:, :] init_rover_orientations(self):
        return self.m_init_rover_orientations
        
    cpdef const double[:, :] rover_positions(self):
        return self.m_rover_positions
        
    cpdef const double[:, :] rover_orientations(self):
        return self.m_rover_orientations
        
    cpdef const double[:] poi_values(self):
        return self.m_poi_values
        
    cpdef const double[:, :] poi_positions(self):
        return self.m_poi_positions

    
    cpdef void set_n_rovers(
            self, 
            Py_ssize_t n_rovers, 
            const double[:, :] init_rover_positions,
            const double[:, :] init_rover_orientations):
        if n_rovers < 0:
            raise ValueError(
                "The number of rovers (n_rovers) must be non-negative. "
                + "The number of rovers received is %d."
                % n_rovers)
        if (init_rover_positions.shape[0] != n_rovers
                or init_rover_positions.shape[1] != 2
                or init_rover_positions.size != n_rovers * 2):
            raise ValueError(
                "The shape of the initial rover position array "
                + "(init_rover_positions) must be (%d, 2) [(n_rovers, 2)]. "
                % n_rovers
                + "The received shape is %s."
                % str(init_rover_positions.shape))
        if (init_rover_orientations.shape[0] != n_rovers
                or init_rover_orientations.shape[1] != 2
                or init_rover_orientations.size != n_rovers * 2):
            raise ValueError(
                "The shape of the initial rover orientation array "
                + "(init_rover_orientations) must be (%d, 2) [(n_rovers, 2)]. "
                % n_rovers
                + "The received shape is %s."
                % str(init_rover_orientations.shape))
            
        self.m_n_rovers = n_rovers
        self.m_init_rover_positions = np.zeros((n_rovers, 2))
        self.m_init_rover_orientations = np.zeros((n_rovers, 2))
        self.m_rover_positions = np.zeros((n_rovers, 2))
        self.m_rover_orientations = np.zeros((n_rovers, 2))
        
        self.m_init_rover_positions[...] = init_rover_positions
        self.m_init_rover_orientations[...] = init_rover_orientations
        self.m_rover_positions[...] = init_rover_positions
        self.m_rover_orientations[...] = init_rover_orientations
        
        
    cpdef void set_n_pois(
            self, 
            Py_ssize_t n_pois, 
            const double[:, :] poi_positions, 
            const double[:] poi_values):
        if n_pois < 0:
            raise ValueError(
                "The number of pois (n_pois) must be non-negative. "
                + "The number of pois received is %d."
                % n_pois)
        if (poi_positions.shape[0] != n_pois
                or poi_positions.shape[1] != 2
                or poi_positions.size != n_pois * 2):
            raise ValueError(
                "The shape of the initial poi position array "
                + "(init_poi_positions) must be (%d, 2) [(n_pois, 2)]. "
                % n_pois
                + "The received shape is %s."
                % str(poi_positions.shape))
        if (poi_positions.shape[0] != n_pois
                or poi_positions.size != n_pois * 2):
            raise ValueError(
                "The shape of the poi values array"
                + "(poi_values) must be (%d,) [(n_pois,)]. "
                % n_pois
                + "The received shape is %s."
                % str(poi_values.shape))
            
        self.m_n_pois = n_pois
        self.m_poi_positions = np.zeros((n_pois, 2))
        self.m_poi_values = np.zeros((n_pois,))
    
        self.m_poi_positions[...] = poi_positions
        self.m_poi_values[...] = poi_values
        
    cpdef void set_init_rover_positions(
            self, 
            const double[:, :] init_rover_positions):
        self.m_init_rover_positions[...] = init_rover_positions
        
    cpdef void set_init_rover_position(
            self, 
            Py_ssize_t rover_id, 
            const double[:] init_rover_position):
        self.m_init_rover_positions[rover_id, ...] = init_rover_position  
    
    cpdef void set_init_rover_orientations(
            self, 
            const double[:, :] init_rover_orientations):
        self.m_init_rover_orientations[...] = init_rover_orientations
        
    cpdef void set_init_rover_orientation(
            self, 
            Py_ssize_t rover_id, 
            const double[:] init_rover_orientation):
        self.m_init_rover_orientations[rover_id, ...] = init_rover_orientation
        
    cpdef void set_rover_positions(
            self, 
            const double[:, :] rover_positions):
        self.m_rover_positions[...] = rover_positions    
        
    cpdef void set_rover_position(
            self, 
            Py_ssize_t rover_id, 
            const double[:] rover_position):
        self.m_rover_positions[rover_id, ...] = rover_position 
        
        
    cpdef void set_rover_orientations(
            self, 
            const double[:, :] rover_orientations):
        self.m_rover_orientations[...] = rover_orientations    
        
    cpdef void set_rover_orientation(
            self, 
            Py_ssize_t rover_id, 
            const double[:] rover_orientation):
        self.m_rover_orientations[rover_id, ...] = rover_orientation
        
    cpdef void set_poi_values(self, const double[:] poi_values):
        self.m_poi_values[...] = poi_values
        
    cpdef void set_poi_value(self, Py_ssize_t poi_id, double poi_value):
        self.m_poi_value[poi_id] = poi_value
        
    cpdef void set_poi_positions(self, const double[:, :] poi_positions):
        self.m_poi_positions[...] = poi_positions
        
    cpdef void set_poi_position(
            self, 
            Py_ssize_t poi_id, 
            const double[:] poi_position):
        self.m_poi_positions[poi_id, ...] = poi_position