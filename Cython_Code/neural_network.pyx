import numpy as np


cdef class NeuralNetwork:
    # Declare variables
    cdef int n_rovers
    cdef int n_inputs
    cdef int n_outputs
    cdef int n_nodes
    cdef int n_weights
    cdef double input_bias
    cdef double hidden_bias
    cdef public double[:, :] weights
    cdef public double[:, :] in_layer
    cdef public double[:, :] hid_layer
    cdef public double[:, :] out_layer

    def __cinit__(self, object p):
        self.n_rovers = int(p.num_rovers)
        self.n_inputs = int(p.num_inputs)
        self.n_outputs = int(p.num_outputs)
        self.n_nodes = int(p.num_nodes)  # Number of nodes in hidden layer
        self.n_weights = int((self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs)
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    cpdef reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    cpdef get_inputs(self, double [:] state_vec, int rov_id):  # Get inputs from state-vector
        """
        Assign inputs from rover sensors to the input layer of the NN
        :param state_vec: Inputs from rover sensors
        :param rov_id: Current rover
        :return: None
        """
        cdef int i
        for i in range(self.n_inputs):
            self.in_layer[rov_id, i] = state_vec[i]

    cpdef get_weights(self, double [:] nn_weights, int rov_id):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :param nn_weights:
        :param rov_id:
        :return: None
        """
        cdef int w

        for w in range(self.n_weights):
            self.weights[rov_id, w] = nn_weights[w]

    cpdef reset_layers(self, int rov_id):  # Clear hidden layers and output layers
        """
        Zeros hidden layer and output layer of NN
        :param rov_id:
        :return: None
        """
        cdef int n

        for n in range(self.n_nodes):
            self.hid_layer[rov_id, n] = 0.0
        for n in range(self.n_outputs):
            self.out_layer[rov_id, n] = 0.0

    cpdef get_outputs(self, int rov_id):
        """
        Run NN to receive rover action outputs
        :param rov_id:
        :return: None
        """
        cdef int count, i, n

        count = 0  # Keeps count of which weight is being applied
        self.reset_layers(rov_id)

        for i in range(self.n_inputs):  # Pass inputs to hidden layer
            for n in range(self.n_nodes):
                self.hid_layer[rov_id, n] += self.in_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for n in range(self.n_nodes):  # Add Biasing Node
            self.hid_layer[rov_id, n] += (self.input_bias * self.weights[rov_id, count])
            count += 1

        for n in range(self.n_nodes):  # Pass hidden layer nodes through activation function
            self.hid_layer[rov_id, n] = self.tanh(self.hid_layer[rov_id, n])

        for i in range(self.n_nodes):  # Pass from hidden layer to output layer
            for n in range(self.n_outputs):
                self.out_layer[rov_id, n] += self.hid_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for n in range(self.n_outputs):  # Add biasing node
            self.out_layer[rov_id, n] += (self.hidden_bias * self.weights[rov_id, count])
            count += 1

        for n in range(self.n_outputs):  # Pass output nodes through activation function
            self.out_layer[rov_id, n] = self.tanh(self.out_layer[rov_id, n])

    cpdef tanh(self, double inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """
        cdef double tanh

        tanh = (2/(1 + np.exp(-2*inp)))-1
        return tanh

    cpdef sigmoid(self, double inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """
        cdef double sig

        sig = 1/(1 + np.exp(-inp))
        return sig

    cpdef run_neural_network(self, double [:] state_vec, double [:] weight_vec, int rover_id):
        """
        Run through NN for given rover
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :param rover_id: Rover identifier
        :return: None
        """
        self.get_inputs(state_vec, rover_id)
        self.get_weights(weight_vec, rover_id)
        self.get_outputs(rover_id)
