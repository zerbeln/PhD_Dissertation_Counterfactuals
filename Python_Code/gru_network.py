import numpy as np
from parameters import parameters


class NeuralNetwork:

    def __init__(self):

        # GRU Properties
        self.n_inputs = int(parameters["n_inputs"])
        self.n_outputs = int(parameters["n_outputs"])
        self.mem_block_size = int(parameters["mem_block_size"])

        # Network Outputs
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))
        self.out_layer_weights = np.mat(np.zeros(self.mem_block_size))
        self.out_bias_weights = np.mat(np.zeros(self.n_outputs))

        # Input Gate
        self.igate_weights = {}
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Block Input
        self.block_weights = {}
        self.block_input = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))

        # Read Gate
        self.rgate_weights = {}
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Write Gate
        self.wgate_weights = {}
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Memory
        self.decoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.encoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.decoder_weights = {}
        self.encoder_weights = {}

    def clear_outputs(self):
        """
        Clears the various outputs of gates (sets them to 0)
        :return:
        """
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """

        # Network Outputs
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))
        self.out_layer_weights = np.mat(np.zeros(self.mem_block_size))  # nx1
        self.out_bias_weights = np.mat(np.zeros(self.n_outputs))  # 2x1

        # Input Gate
        self.igate_weights = {}
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Block Input
        self.block_input = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))
        self.block_weights = {}

        # Read Gate
        self.rgate_weights = {}
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Write Gate
        self.wgate_weights = {}
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Memory
        self.decoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.encoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.decoder_weights = {}
        self.encoder_weights = {}

    def get_weights(self, weights):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :return: None
        """

        # Output weights
        self.out_bias_weights = np.mat(weights["b_out"])
        self.out_layer_weights = np.mat(weights["p_out"])

        # Input gate weights
        n_mat = np.mat(weights["n_igate"])
        self.igate_weights["K"] = np.reshape(np.mat(weights["k_igate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.igate_weights["R"] = np.reshape(np.mat(weights["r_igate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.igate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.igate_weights["b"] = np.reshape(np.mat(weights["b_igate"]), [self.mem_block_size, 1])  # nx1

        # Block Input
        n_mat = np.mat(weights["n_block"])
        self.block_weights["K"] = np.reshape(np.mat(weights["k_block"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.block_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.block_weights["b"] = np.reshape(np.mat(weights["b_block"]), [self.mem_block_size, 1])  # nx1

        # Read gate weights
        n_mat = np.mat(weights["n_rgate"])
        self.rgate_weights["K"] = np.reshape(np.mat(weights["k_rgate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.rgate_weights["R"] = np.reshape(np.mat(weights["r_rgate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.rgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.rgate_weights["b"] = np.reshape(np.mat(weights["b_rgate"]), [self.mem_block_size, 1])  # nx1

        # Write Gate Weights
        n_mat = np.mat(weights["n_wgate"])
        self.wgate_weights["K"] = np.reshape(np.mat(weights["k_wgate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.wgate_weights["R"] = np.reshape(np.mat(weights["r_wgate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.wgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.wgate_weights["b"] = np.reshape(np.mat(weights["b_wgate"]), [self.mem_block_size, 1])  # nx1

        # Memory Weights
        n_mat_dec = np.mat(weights["n_dec"])
        n_mat_enc = np.mat(weights["z_enc"])
        self.decoder_weights["N"] = np.reshape(n_mat_dec, [self.mem_block_size, self.mem_block_size])  # nxn
        self.decoder_weights["b"] = np.reshape(np.mat(weights["b_dec"]), [self.mem_block_size, 1])  # nx1
        self.encoder_weights["Z"] = np.reshape(n_mat_enc, [self.mem_block_size, self.mem_block_size])  # nxn
        self.encoder_weights["b"] = np.reshape(np.mat(weights["b_enc"]), [self.mem_block_size, 1])  # nx1

    def run_input_gate(self, state_vec, mem_block):
        """
        Process sensor inputs through input gate
        :param mem:
        :param state_vec:
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.igate_weights["b"]  # nx1

        Ki_x = np.dot(self.igate_weights["K"], x)  # nx8 * 8x1 = nx1
        Ri_y = np.dot(self.igate_weights["R"], y)  # nx2 * 2x1 = nx1
        Ni_m = np.dot(self.igate_weights["N"], m)  # nxn * nx1 = nx1

        self.igate_outputs = Ki_x + Ri_y + Ni_m + b  # nx1
        for i in range(self.mem_block_size):
            self.igate_outputs[i, 0] = self.sigmoid(self.igate_outputs[i, 0])

    def run_read_gate(self, state_vec, mem_block):
        """
        Process read gate information
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.rgate_weights["b"]  # nx1

        Kr_x = np.dot(self.rgate_weights["K"], x)  # nx8 * 8x1
        Rr_y = np.dot(self.rgate_weights["R"], y)  # nx2 * 2x1
        Nr_m = np.dot(self.rgate_weights["N"], m)  # nxn * 1x1

        self.rgate_outputs = Kr_x + Rr_y + Nr_m + b  # nx1
        for i in range(self.mem_block_size):
            self.rgate_outputs[i, 0] = self.sigmoid(self.rgate_outputs[i, 0])

    def run_write_gate(self, state_vec, mem_block):
        """
        Process write gate
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.wgate_weights["b"]  # nx1

        Kw_x = np.dot(self.wgate_weights["K"], x)  # nx8 * 8x1
        Rw_x = np.dot(self.wgate_weights["R"], y)  # nx2 * 2x1
        Nw_x = np.dot(self.wgate_weights["N"], m)  # nxn * nx1

        self.wgate_outputs = Kw_x + Rw_x + Nw_x + b  # nx1
        for i in range(self.mem_block_size):
            self.wgate_outputs[i, 0] = self.sigmoid(self.wgate_outputs[i, 0])

    def create_block_inputs(self, state_vec, mem_block):
        """
        Create the input layer for the block (feedforward network)
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.block_weights["b"]  # nx1

        Kp_x = np.dot(self.block_weights["K"], x)  # nx8 * 8x1
        Np_m = np.dot(self.block_weights["N"], m)  # nxn * nx1

        self.block_input = Kp_x + Np_m + b  # nx1
        for i in range(self.mem_block_size):
            self.block_input[i, 0] = self.sigmoid(self.block_input[i, 0])

    def memory_decoder(self, mem_block):
        """
        Decode memory for hidden activation
        :param mem_block:
        :return:
        """

        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])
        Nd_m = np.dot(self.decoder_weights["N"], m)  # nxn * nx1
        b = self.decoder_weights["b"]  # nx1

        self.decoded_memory = Nd_m + b  # nx1
        for i in range(self.mem_block_size):
            self.decoded_memory[i, 0] = self.tanh(self.decoded_memory[i, 0])

    def memory_encoder(self):
        """
        Encode memory for update
        :return:
        """
        b = self.encoder_weights["b"]  # nx1
        Z_h = np.dot(self.encoder_weights["Z"], self.block_output)  # nxn * nx1

        self.encoded_memory = Z_h + b  # nx1
        for i in range(self.mem_block_size):
            self.encoded_memory[i, 0] = self.tanh(self.encoded_memory[i, 0])

    def hidden_activation(self):
        """
        Run NN to receive rover action outputs
        :return: None
        """

        r_d = np.multiply(self.rgate_outputs, self.decoded_memory)  # nx1
        p_i = np.multiply(self.block_input, self.igate_outputs)  # nx1

        self.block_output = r_d + p_i  # nx1

        self.out_layer = np.dot(self.out_layer_weights, self.block_output) + self.out_bias_weights  # 1x1

        for v in range(self.n_outputs):
            self.out_layer[0, v] = self.tanh(self.out_layer[0, v])

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

    def run_neural_network(self, state_vec, mem_block):
        """
        Run through NN for given rover
        :param mem_block:
        :param state_vec:
        :return: None
        """
        self.run_input_gate(state_vec, mem_block)
        self.run_read_gate(state_vec, mem_block)
        self.create_block_inputs(state_vec, mem_block)
        self.memory_decoder(mem_block)
        self.hidden_activation()
        self.run_write_gate(state_vec, mem_block)
        self.memory_encoder()
        self.prev_out_layer = self.out_layer.copy()