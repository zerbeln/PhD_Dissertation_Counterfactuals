import numpy as np
from AADI_RoverDomain.parameters import Parameters as p


class NeuralNetwork:

    def __init__(self):
        self.n_inputs = p.num_inputs
        self.n_outputs = p.num_outputs
        self.n_nodes = p.num_nodes  # Number of nodes in hidden layer
        self.n_weights = (self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.weights = np.zeros((p.num_rovers, self.n_weights))
        self.in_layer = np.zeros((p.num_rovers, self.n_inputs))
        self.hid_layer = np.zeros((p.num_rovers, self.n_nodes))
        self.out_layer = np.zeros((p.num_rovers, self.n_outputs))

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """
        self.weights = np.zeros((p.num_rovers, self.n_weights))
        self.in_layer = np.zeros((p.num_rovers, self.n_inputs))
        self.hid_layer = np.zeros((p.num_rovers, self.n_nodes))
        self.out_layer = np.zeros((p.num_rovers, self.n_outputs))

    def get_inputs(self, state_vec, rov_id):  # Get inputs from state-vector
        """
        Assign inputs from rover sensors to the input layer of the NN
        :param state_vec: Inputs from rover sensors
        :param rov_id: Current rover
        :return: None
        """
        for i in range(self.n_inputs):
            self.in_layer[rov_id, i] = state_vec[i]

    def get_weights(self, nn_weights, rov_id):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :param nn_weights:
        :param rov_id:
        :return: None
        """

        for w in range(self.n_weights):
            self.weights[rov_id, w] = nn_weights[w]

    def reset_layers(self, rov_id):  # Clear hidden layers and output layers
        """
        Zeros hidden layer and output layer of NN
        :param rov_id:
        :return: None
        """
        for i in range(self.n_nodes):
            self.hid_layer[rov_id, i] = 0.0

        for j in range(self.n_outputs):
            self.out_layer[rov_id, j] = 0.0

    def get_outputs(self, rov_id):
        """
        Run NN to receive rover action outputs
        :param rov_id:
        :return: None
        """
        count = 0  # Keeps count of which weight is being applied
        self.reset_layers(rov_id)

        # for i in range(self.n_inputs):
        #     self.in_layer[rov_id, i] = self.tanh(self.in_layer[rov_id, i])

        for i in range(self.n_inputs):  # Pass inputs to hidden layer
            for j in range(self.n_nodes):
                self.hid_layer[rov_id, j] += self.in_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for j in range(self.n_nodes):  # Add Biasing Node
            self.hid_layer[rov_id, j] += (self.input_bias * self.weights[rov_id, count])
            count += 1

        for i in range(self.n_nodes):  # Pass through sigmoid
            self.hid_layer[rov_id, i] = self.tanh(self.hid_layer[rov_id, i])

        for i in range(self.n_nodes):  # Pass from hidden layer to output layer
            for j in range(self.n_outputs):
                self.out_layer[rov_id, j] += self.hid_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for j in range(self.n_outputs):  # Add biasing node
            self.out_layer[rov_id, j] += (self.hidden_bias * self.weights[rov_id, count])
            count += 1

        for i in range(self.n_outputs):  # Pass through sigmoid
            self.out_layer[rov_id, i] = self.tanh(self.out_layer[rov_id, i])

    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """
        tanh = (2/(1 + np.exp(-2*inp)))-1
        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """
        sig = 1/(1 + np.exp(-inp))
        return sig

    def run_neural_network(self, rover_input, weight_vec, rover_id):
        """
        Run through NN for given rover
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :param rover_id: Rover identifier
        :return: None
        """
        self.get_inputs(rover_input, rover_id)
        self.get_weights(weight_vec, rover_id)
        self.get_outputs(rover_id)
