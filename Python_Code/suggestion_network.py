import numpy as np


class SuggestionNetwork:
    def __init__(self, n_in, n_out, n_hid):
        self.n_inputs = n_in
        self.n_outputs = n_out
        self.n_hnodes = n_hid  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def get_inputs(self, suggest_inputs):  # Get inputs from state-vector
        """
        Transfer state information to the neuro-controller
        :return:
        """

        for i in range(self.n_inputs):
            self.input_layer[i, 0] = suggest_inputs[i]

    def get_weights(self, weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        :param nn_weights: Dictionary of network weights received from the CCEA
        :return:
        """
        self.weights["Layer1"] = np.reshape(np.mat(weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(weights["b2"]), [self.n_outputs, 1])

    def get_outputs(self):
        """
        Run NN to generate outputs
        :return:
        """
        outputs = np.zeros(self.n_outputs)
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        for i in range(self.n_hnodes):
            self.hidden_layer[i, 0] = self.sigmoid(self.hidden_layer[i, 0])

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        for i in range(self.n_outputs):
            self.output_layer[i, 0] = self.sigmoid(self.output_layer[i, 0])
            outputs[i] = self.output_layer[i, 0]

        return outputs

    def run_network(self, s_input):
        """
        Run the entire network from one function call
        """
        self.get_inputs(s_input)
        nn_outputs = self.get_outputs()

        return nn_outputs

    # Activation Functions -------------------------------------------------------------------------------------------
    def tanh(self, inp):  # Tanh function as activation function
        """
        tanh neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        tanh = (2 / (1 + np.exp(-2 * inp))) - 1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        sigmoid neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        if inp >= 0:
            sig = 1 / (1 + np.exp(-inp))
            return sig
        else:
            sig = 1 / (1 + np.exp(inp))
            return sig
