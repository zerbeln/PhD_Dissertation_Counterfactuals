import numpy as np

class rnn:

    def __init__(self, num_inputs, num_outputs, num_hnodes, mean=0, std=1):
        self.n_inputs = num_inputs + num_outputs  # Outputs added for recursion
        self.n_outputs = num_outputs
        self.n_hnodes = num_hnodes
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.in_layer = np.zeros(self.n_inputs, dtype=np.float64)
        self.hid_layer = np.zeros(self.n_hnodes, dtype=np.float64)
        self.out_layer = np.zeros(self.n_outputs, dtype=np.float64)
        self.previous_outputs = np.zeros(self.n_outputs, dtype=np.float64)

        # Weight matrices
        self.total_weights = int((self.n_inputs + 1) * self.n_hnodes + (self.n_hnodes + 1) * self.n_outputs)
        self.n_hidden_weights = (self.n_inputs + 1) * self.n_hnodes  # +1 is to include biasing node
        self.n_output_weights = (self.n_hnodes + 1) * self.n_outputs  # +1 is to include biasing node
        self.inp_weights = np.zeros(self.n_hidden_weights, dtype=np.float64)
        self.out_weights = np.zeros(self.n_output_weights, dtype=np.float64)

        # Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_inputs + 1)))
        self.w_inpgate = np.mat(np.reshape(self.w_inpgate, (num_hnodes, (num_inputs + 1))))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_outputs + 1)))
        self.w_rec_inpgate = np.mat(np.reshape(self.w_rec_inpgate, (num_hnodes, (num_outputs + 1))))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_hnodes + 1)))
        self.w_mem_inpgate = np.mat(np.reshape(self.w_mem_inpgate, (num_hnodes, (num_hnodes + 1))))

        # Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, num_hnodes * (num_inputs + 1)))
        self.w_inp = np.mat(np.reshape(self.w_inp, (num_hnodes, (num_inputs + 1))))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, num_hnodes * (num_outputs + 1)))
        self.w_rec_inp = np.mat(np.reshape(self.w_rec_inp, (num_hnodes, (num_outputs + 1))))

        # Forget gate
        self.w_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_inputs + 1)))
        self.w_forgetgate = np.mat(np.reshape(self.w_forgetgate, (num_hnodes, (num_inputs + 1))))
        self.w_rec_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_outputs + 1)))
        self.w_rec_forgetgate = np.mat(np.reshape(self.w_rec_forgetgate, (num_hnodes, (num_outputs + 1))))
        self.w_mem_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes * (num_hnodes + 1)))
        self.w_mem_forgetgate = np.mat(np.reshape(self.w_mem_forgetgate, (num_hnodes, (num_hnodes + 1))))

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """
        self.weights = np.zeros(self.total_weights, dtype=np.float64)
        self.in_layer = np.zeros(self.n_inputs, dtype=np.float64)
        self.hid_layer = np.zeros(self.n_hnodes, dtype=np.float64)
        self.out_layer = np.zeros(self.n_outputs, dtype=np.float64)
        self.previous_outputs = np.zeros(self.n_outputs, dtype=np.float64)


    def get_inputs(self, state_vec):  # Get inputs from state-vector
        """
        Create input layer for neural network as concatentation of previous outputs and current sensor inputs
        :param state_vec: Inputs from rover sensors
        :return: None
        """

        self.in_layer = np.concatenate((state_vec, self.previous_outputs))

    def get_weights(self, nn_weights):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :param nn_weights:
        :param rov_id:
        :return: None
        """

        weight_count = 0
        for w in range(self.total_weights):
            if w < self.n_hidden_weights:
                self.inp_weights[w] = nn_weights[w]
            else:
                self.out_weights[weight_count] = nn_weights[w]
                weight_count += 1

    def reset_layers(self):  # Clear hidden layers and output layers
        """
        Zeros hidden layer and output layer of NN
        :param rov_id:
        :return: None
        """

        for n in range(self.n_hnodes):
            self.hid_layer[n] = 0.0
        for n in range(self.n_outputs):
            self.out_layer[n] = 0.0

    def get_outputs(self):
        """
        Run NN to receive rover action outputs
        :param sweep:
        :return: None
        """
        self.reset_layers()

        # Reshape weight arrays into a matrix for matrix multiplication
        ih_weights = np.reshape(self.inp_weights, [self.n_inputs + 1, self.n_hnodes])
        ho_weights = np.reshape(self.out_weights, [self.n_hnodes + 1, self.n_outputs])

        self.hid_layer = np.dot(self.in_layer, ih_weights)
        self.hid_layer = np.append(self.hid_layer, self.hidden_bias)  # Append biasing node to hidden layer


        for n in range(self.n_hnodes):  # Pass hidden layer nodes through activation function
            self.hid_layer[n] = self.tanh(self.hid_layer[n])

        self.out_layer = np.dot(self.hid_layer, ho_weights)

        for n in range(self.n_outputs):  # Pass output nodes through activation function
            self.out_layer[n] = self.tanh(self.out_layer[n])

        self.previous_outputs = self.out_layer.copy()

    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """

        tanh = (2 / (1 + np.exp(-2 * inp))) - 1
        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """

        sig = 1 / (1 + np.exp(-inp))
        return sig

    def run_neural_network(self, state_vec, weight_vec):
        """
        Run through NN for given rover
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :param rover_id: Rover identifier
        :return: None
        """
        self.get_inputs(state_vec)
        self.get_weights(weight_vec)
        self.get_outputs()