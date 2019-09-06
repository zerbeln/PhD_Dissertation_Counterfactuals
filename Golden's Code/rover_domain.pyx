
cdef class RoverDomain:
    
    def __init__(self):
        self.m_state = State()
        self.m_rover_observations_getter = RoverObservationsGetter()
        self.m_rover_actions_simulator = RoverActionsSimulator()
        self.m_evaluator = Evaluator()
        self.test_sync()

    cpdef State state(self):
        return self.m_state
        
    cpdef RoverObservationsGetter rover_observations_getter(self):
        return self.m_rover_actions_simulator
        
    cpdef RoverActionsSimulator rover_actions_simulator(self):
        return self.m_rover_actions_simulator
        
    cpdef Evaluator evaluator(self):
        return self.m_evaluator
    
    cpdef void set_state(self, State state):
        self.m_state = state
        self.test_sync()
        
    cpdef void set_rover_observations_getter(
            self, 
            RoverObservationsGetter rover_observations_getter):
        self.m_rover_observation = rover_observations_getter
        self.test_sync()
        
    cpdef void set_rover_actions_simulator(
            self, 
            RoverActionsSimulator rover_actions_simulator):
        self.m_rover_actions_simulator = rover_actions_simulator
        self.test_sync()
        
    cpdef void set_evaluator(self, Evaluator evaluator):
        self.m_evaluator = evaluator
        self.test_sync()
    
    cpdef void test_sync(self):
        # Sync n_pois, n_rovers, 
        cdef Py_ssize_t n_rovers, n_pois, test_n_rovers, test_n_pois
        
        n_rovers = self.m_state.n_rovers()
        n_pois = self.m_state.n_pois()
        
        test_n_rovers = self.m_rover_observations_getter.n_rovers()
        if n_rovers != test_n_rovers:
            raise ValueError(
                "The rover_observations_getter's number of rover (n_rovers) of "
                + "%d does not match the state's n_rovers of %d"
                %(test_n_rovers, n_rovers))
    
        test_n_rovers = self.m_evaluator.n_rovers()
        test_n_pois = self.m_evaluator.n_pois()
        if n_rovers != test_n_rovers:
            raise ValueError(
                "The evaluator's number of rover (n_rovers) of "
                + "%d does not match the state's n_rovers of %d"
                %(test_n_rovers, n_rovers))
        if n_pois != test_n_pois:
            raise ValueError(
                "The evaluator's number of POIs (n_pois) of "
                + "%d does not match the state's n_pois of %d"
                %(test_n_pois, n_pois))
    
    
    # Composite Non-Accessor Methods
    cpdef void reset(self):
        self.m_state.reset()
        
    cpdef void simulate_rover_actions(
            self, 
            const double[:, :] rover_actions,
            Py_ssize_t step_id):
        self.m_evaluator.update_rover_position_histories(
            step_id, 
            self.m_state.rover_positions())
        self.m_evaluator.update_rover_action_histories(
            step_id, 
            rover_actions)
        self.m_rover_actions_simulator.simulate_rover_actions(
            self.m_state,
            rover_actions)
            



    
    # Compositie Getters
    cpdef Py_ssize_t n_rovers(self):
        return self.m_state.n_rovers()
        
    cpdef Py_ssize_t n_pois(self):
        return self.m_state.n_pois()
        
    cpdef Py_ssize_t n_rover_observation_sections(self):
        return self.m_rover_observations_getter.n_rover_observation_sections()

    cpdef Py_ssize_t n_steps(self):
        return self.m_evaluator.n_steps()
        
    cpdef Py_ssize_t n_req(self):
        return self.m_evaluator.n_req()
    
    cpdef const double[:, :] init_rover_positions(self):
        return self.m_state.init_rover_positions()
        
    cpdef const double[:, :] init_rover_orientations(self):
        return self.m_state.init_rover_orientations()
        
    cpdef const double[:, :] rover_positions(self):
        return self.m_state.rover_positions()
        
    cpdef const double[:, :] rover_orientations(self):
        return self.m_state.rover_orientations()
        
    cpdef const double[:] poi_values(self):
        return self.m_state.poi_values()
        
    cpdef const double[:, :] poi_positions(self):
        return self.m_state.poi_positions()
        
    cpdef const double[:, :] rover_observations(self):
        return self.m_rover_observations_getter.rover_observations(
            self.m_state)
    
    cpdef const double[:] rover_evals(self):
        return self.m_evaluator.rover_evals(self.m_state)
        
    cpdef const double[:, :, :] rover_position_histories(self):
        return self.m_evaluator.rover_position_histories()
    
    cpdef double min_dist(self):
        return self.m_rover_observations_getter.min_dist()
        
    cpdef double eval(self):
        return self.m_evaluator.eval(self.m_state)
        
    cpdef double interaction_dist(self):
        return self.m_evaluator.interaction_dist()
    
    
    # Composite Setters
    cpdef void set_n_rovers(
            self, 
            Py_ssize_t n_rovers, 
            const double[:, :] init_rover_positions,
            const double[:, :] init_rover_orientations):
        self.m_state.set_n_rovers(
            n_rovers, 
            init_rover_positions,
            init_rover_orientations)
        self.m_rover_observations_getter.set_n_rovers(n_rovers)
        self.m_evaluator.set_n_rovers(n_rovers)
        
    cpdef void set_n_pois(
            self, 
            Py_ssize_t n_pois, 
            const double[:, :] poi_positions, 
            const double[:] poi_values):
        self.m_state.set_n_pois(n_pois, poi_positions, poi_values)
        self.m_evaluator.set_n_pois(n_pois)
        
    cpdef void set_n_rover_observation_sections(
            self, 
            Py_ssize_t n_rover_observation_sections):
        self.m_rover_observations_getter.set_n_rover_observation_sections(
            n_rover_observation_sections)

    cpdef void set_n_steps(self, Py_ssize_t n_steps):
        self.m_evaluator.set_n_steps(n_steps)
        
    cpdef void set_n_req(self, Py_ssize_t n_req):
        self.m_evaluator.set_n_req(n_req)
    
    cpdef void set_init_rover_positions(
            self, 
            const double[:, :] init_rover_positions):
        self.m_state.set_init_rover_positions(init_rover_positions)
        
    cpdef void set_init_rover_position(
            self, 
            Py_ssize_t rover_id, 
            const double[:] init_rover_position):
        self.m_state.set_init_rover_position(rover_id, init_rover_position)
        
    cpdef void set_init_rover_orientations(
            self, 
            const double[:, :] init_rover_orientations):
        self.m_state.set_init_rover_orientations(init_rover_orientations)
        
    cpdef void set_init_rover_orientation(
            self, 
            Py_ssize_t rover_id, 
            const double[:] init_rover_orientation):
        self.m_state.set_init_rover_orientation(
            rover_id, 
            init_rover_orientation)
        
    cpdef void set_rover_positions(
            self, 
            const double[:, :] rover_positions):
        self.m_state.set_rover_positions(rover_positions)
        
    cpdef void set_rover_position(
            self, 
            Py_ssize_t rover_id, 
            const double[:] rover_position):
        self.m_state.set_rover_position(rover_id, rover_position)
        
    cpdef void set_rover_orientations(
            self, 
            const double[:, :] rover_orientations):
        self.m_state.set_rover_orientations(rover_orientations)
        
    cpdef void set_rover_orientation(
            self, 
            Py_ssize_t rover_id, 
            const double[:] rover_orientation):
        self.m_state.set_rover_orientation(rover_id, rover_orientation)
        
    cpdef void set_poi_values(self, const double[:] poi_values):
        self.m_state.set_poi_values(poi_values)
        
    cpdef void set_poi_value(self, Py_ssize_t poi_id, double poi_value):
        self.m_state.set_poi_value(poi_id, poi_value)
        
    cpdef void set_poi_positions(self, const double[:, :] poi_positions):
        self.m_state.set_poi_positions(poi_positions)
        
    cpdef void set_poi_position(
            self, 
            Py_ssize_t poi_id, 
            const double[:] poi_position):
        self.m_state.set_poi_position(poi_id, poi_position)
        
    cpdef void set_min_dist(self, double min_dist):
        self.m_rover_observations_getter.set_min_dist(min_dist)

    cpdef void set_interaction_dist(self, double interaction_dist):
        self.m_evaluator.set_interaction_dist(interaction_dist)
    
