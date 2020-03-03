import math
#import MultiNEAT as NEAT
import numpy as np
from scipy.special import expit

class Quasi_GRU:
    def __init__(self, num_input, num_hnodes, num_output, mean=0, std=1):
        self.arch_type = 'quasi_gru'
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes

        # Adaptive components (plastic with network running)
        self.last_output = np.mat(np.zeros(num_output)).transpose()
        self.memory_cell = np.mat(np.random.normal(mean, std, num_hnodes)).transpose() #Memory Cell

        # Banks for adaptive components, that can be used to reset
        # self.bank_last_output = self.last_output[:]
        self.bank_memory_cell = self.memory_cell[:]  # Memory Cell

        ## WEIGHT MATRICES --------------------------------------------------------------------------------------------
        # Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inpgate = np.mat(np.reshape(self.w_inpgate, (num_hnodes, (num_input + 1))))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inpgate = np.mat(np.reshape(self.w_rec_inpgate, (num_hnodes, (num_output + 1))))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_inpgate = np.mat(np.reshape(self.w_mem_inpgate, (num_hnodes, (num_hnodes + 1))))

        # Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inp = np.mat(np.reshape(self.w_inp, (num_hnodes, (num_input + 1))))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inp = np.mat(np.reshape(self.w_rec_inp, (num_hnodes, (num_output + 1))))

        # Forget gate
        self.w_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_forgetgate = np.mat(np.reshape(self.w_forgetgate, (num_hnodes, (num_input + 1))))
        self.w_rec_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_forgetgate = np.mat(np.reshape(self.w_rec_forgetgate, (num_hnodes, (num_output + 1))))
        self.w_mem_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_forgetgate = np.mat(np.reshape(self.w_mem_forgetgate, (num_hnodes, (num_hnodes + 1))))

        # Output weights
        self.w_output = np.mat(np.random.normal(mean, std, num_output*(num_hnodes + 1)))
        self.w_output = np.mat(np.reshape(self.w_output, (num_output, (num_hnodes + 1)))) #Reshape the array to the weight matrix

    def linear_combination(self, w_matrix, layer_input):  # Linear combine weights with inputs
        return np.dot(w_matrix, layer_input)  # Linear combination of weights and inputs

    def relu(self, layer_input):  # Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def sigmoid(self, layer_input):  # Sigmoid transform

        # Just make sure the sigmoid does not explode and cause a math error
        p = layer_input.A[0][0]
        if p > 700:
            ans = 0.999999
        elif p < -700:
            ans = 0.000001
        else:
            ans =  1 / (1 + math.exp(-p))

        return ans

    def fast_sigmoid(self, layer_input):  # Sigmoid transform
        layer_input = expit(layer_input)
        # for i in layer_input: i[0] = i / (1 + math.fabs(i))
        return layer_input

    def softmax(self, layer_input):  # Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def format_input(self, input, add_bias=True):  # Formats and adds bias term to given input at the end
        if add_bias:
            input = np.concatenate((input, [1.0]))

        return np.mat(input)

    def format_memory(self, memory):
        ig = np.mat([1])

        return np.concatenate((memory, ig))

    # Memory_write gate
    def feedforward(self, input):  # Feedforwards the input and computes the forward pass of the network
        self.input = self.format_input(input).transpose()  # Format and add bias term at the end
        last_memory = self.format_memory(self.memory_cell)
        last_output = self.format_memory(self.last_output)

        # Input gate
        ig_1 = self.linear_combination(self.w_inpgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_inpgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_inpgate, last_memory)
        input_gate_out = ig_1 + ig_2 + ig_3
        input_gate_out = self.fast_sigmoid(input_gate_out)

        # Input processing
        ig_1 = self.linear_combination(self.w_inp, self.input)
        ig_2 = self.linear_combination(self.w_rec_inp, last_output)
        block_input_out = ig_1 + ig_2
        block_input_out = self.fast_sigmoid(block_input_out)

        # Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        # Forget Gate
        ig_1 = self.linear_combination(self.w_forgetgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_forgetgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_forgetgate, last_memory)
        forget_gate_out = ig_1 + ig_2 + ig_3
        forget_gate_out = self.fast_sigmoid(forget_gate_out)

        # Memory Output
        memory_output = np.multiply(forget_gate_out, self.memory_cell)

        # Update memory Cell
        self.memory_cell = memory_output + input_out

        # Compute final output
        new_mem = self.format_memory(self.memory_cell)
        self.last_output = self.linear_combination(self.w_output, new_mem)
        self.last_output = self.fast_sigmoid(self.last_output)

        return np.array(self.last_output).tolist()

    def reset_bank(self):
        # self.last_output = self.bank_last_output[:] #last output
        self.last_output *= 0  # last output
        self.memory_cell = self.bank_memory_cell[:] #Memory Cell

    def set_bank(self):
        # self.bank_last_output = self.last_output[:]  # last output
        self.bank_memory_cell = self.memory_cell[:]  # Memory Cell

    def get_weights(self):
        #TODO NOT OPERATIONAL
        w1 = np.array(self.w_01).flatten().copy()
        w2 = np.array(self.w_12).flatten().copy()
        weights = np.concatenate((w1, w2))

        return weights

    def set_weights(self, weights):
        # Input gates
        start = 0; end = self.num_hnodes*(self.num_input + 1)
        w_inpgate = weights[start:end]
        self.w_inpgate = np.mat(np.reshape(w_inpgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inpgate = weights[start:end]
        self.w_rec_inpgate = np.mat(np.reshape(w_rec_inpgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_inpgate = weights[start:end]
        self.w_mem_inpgate = np.mat(np.reshape(w_mem_inpgate, (self.num_hnodes, (self.num_hnodes + 1))))

        # Block Inputs
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_inp = weights[start:end]
        self.w_inp = np.mat(np.reshape(w_inp, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inp = weights[start:end]
        self.w_rec_inp = np.mat(np.reshape(w_rec_inp, (self.num_hnodes, (self.num_output + 1))))

        # Forget Gates
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_forgetgate = weights[start:end]
        self.w_forgetgate = np.mat(np.reshape(w_forgetgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_forgetgate = weights[start:end]
        self.w_rec_forgetgate = np.mat(np.reshape(w_rec_forgetgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_forgetgate = weights[start:end]
        self.w_mem_forgetgate = np.mat(np.reshape(w_mem_forgetgate, (self.num_hnodes, (self.num_hnodes + 1))))

        # Output weights
        start = end; end += self.num_output*(self.num_hnodes + 1)
        w_output= weights[start:end]
        self.w_output = np.mat(np.reshape(w_output, (self.num_output, (self.num_hnodes + 1))))

        # Memory Cell (prior)
        start = end; end += self.num_hnodes
        memory_cell= weights[start:end]
        self.memory_cell = np.mat(memory_cell).transpose()
