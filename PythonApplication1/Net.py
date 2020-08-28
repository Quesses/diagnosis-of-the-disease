from Neuron import Neuron

class Net:

    def __init__(self, topology):
        num_of_layers = len(topology)
        self.layers = []
        self.average_error = 0
        self.average_errors_history = []
        self.training_counter = 0
        for layer_num in range(num_of_layers):

            self.layers.append([])
            number_of_outputs = 0 if layer_num == num_of_layers - 1 else topology[layer_num+1]

            for neuron_index in range(topology[layer_num]+1):
                self.layers[-1].append(Neuron(number_of_outputs, neuron_index))

            self.layers[-1][-1].output_value = 1


    def getErrors(self):
        return self.average_errors_history

    def feed_forward(self, input_values):

        if len(input_values) != len(self.layers[0])-1:
            raise Exception("Invalid input")

        #setting input values
        for neuron_index in range(len(input_values)):
            self.layers[0][neuron_index].output_value = input_values[neuron_index]

        #feeding forward
        for layer_num in range(1, len(self.layers)):
            previous_layer = self.layers[layer_num -1]
            for neuron_index in range(len(self.layers[layer_num]) - 1):
                self.layers[layer_num][neuron_index].feed_forward(previous_layer)


    def back_prop(self, target_values):

        output_layer = self.layers[-1]
        for n in range(len(output_layer) - 1):
            delta = target_values[n] - output_layer[n].output_value
            self.average_error += delta**2
            pass
        self.training_counter+=1


        previus_layer = self.layers[-2]
        for n in range(len(output_layer) - 1):
            output_layer[n].calc_output_gradients(target_values[n], previus_layer)
            pass

        for layer_num in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[layer_num]
            next_layer = self.layers[layer_num + 1]
            previus_layer = self.layers[layer_num - 1]
            for n in range(len(hidden_layer)):
                hidden_layer[n].calc_hidden_gradients(previus_layer, next_layer)

    def update_weights(self):

        error = self.average_error / 2
        self.average_errors_history.append(error/self.training_counter)
        self.training_counter = 0
        self.average_error = 0

        for layer_num in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_num]
            previous_layer = self.layers[layer_num - 1]
            for n in range(len(layer)):
                layer[n].update_output_weights(previous_layer)
        
        return error
       


    def get_result(self):
        result = []
        for n in range(len(self.layers[-1])-1):
            result.append(self.layers[-1][n].output_value)
        pass
        return result


