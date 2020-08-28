from random import uniform
import numpy as np

class Neuron:
    
    def __init__(self, number_of_connections, neuron_index):
        self.neuron_index = neuron_index
        self.transfer_function_input = 0
        self.output_value = 0
        self.eta = 0.01
        self.alpha = 0.9
        self.gradient = 0
        self.output_weights = []
        for connection in range(number_of_connections + 1):
            self.output_weights.append({"weight":self.random_weight(), "delta_weight":0, "gradient":0})


    def feed_forward(self, previous_layer):
        summed_values = 0.0
        for neuron_index in range(len(previous_layer)):
            summed_values += previous_layer[neuron_index].output_value * \
                previous_layer[neuron_index].output_weights[self.neuron_index]['weight']
            #print([neuron_index, previous_layer[neuron_index].output_weights[self.neuron_index]['weight'], previous_layer[neuron_index].output_value])

        self.transfer_function_input = summed_values
        self.output_value = self.transfer_function(summed_values)


    def sumDOW(self, next_layer):
        sum = 0.0
        for n in range(len(next_layer)):
            sum += self.output_weights[n]['weight'] * next_layer[n].gradient * Neuron.transfer_function_derivative(self.transfer_function_input)

        return sum 

    def calc_output_gradients(self, target_val, prev_layer):
        delta = self.output_value - target_val
        self.gradient = delta * Neuron.transfer_function_derivative(self.transfer_function_input)
        for n in range(len(prev_layer)):
            prev_layer[n].output_weights[self.neuron_index]['gradient'] += self.gradient * prev_layer[n].output_value


    def calc_hidden_gradients(self, prev_layer, next_layer):
        self.gradient = self.sumDOW(next_layer)
        for n in range(len(prev_layer)):
            prev_layer[n].output_weights[self.neuron_index]['gradient'] = self.gradient * prev_layer[n].output_value


    def update_output_weights(self, prev_layer):
        for n in range(len(prev_layer)):
            neuron = prev_layer[n]
            old_delta_weight = neuron.output_weights[self.neuron_index]['delta_weight']
            new_delta_weight = -self.eta * neuron.output_weights[self.neuron_index]['gradient'] + self.alpha * old_delta_weight
            neuron.output_weights[self.neuron_index]['gradient'] = 0

            neuron.output_weights[self.neuron_index]['weight'] += new_delta_weight
            neuron.output_weights[self.neuron_index]['delta_weight'] = new_delta_weight


  
    @staticmethod
    def random_weight():
        return uniform(0, 1)

    @staticmethod
    def transfer_function(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def transfer_function_derivative(x):
        return Neuron.transfer_function(x)*(1.0 - Neuron.transfer_function(x))

