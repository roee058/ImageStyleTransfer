import numpy as np
import tensorflow as tf

'''
CNN (Convolutional Neural Network) Class will perform the general functionality of activation and output return.
'''
class CNN(object):

    def __init__(self):
        pass

    '''
    Will perform full activation of the network and will return the matching output by the input layers names list.
    @param image:             RGB image [batch, IMAGE_SIZE, IMAGE_SIZE, 3]
    @param layers_names_list: List of the requested layers output
    '''
    def activate_network(self, image, layers_names_list):
        layers = [self.layers[layer_name] for layer_name in layers_names_list]
        with tf.Session() as session:
            session.run(self.network_input.assign(image))
            results_array = session.run(layers)
        return results_array

    '''
    Will return list of output tensors from the requested layers after the non-linear operation
    @param layer_names_list - List of names which will return their tensor
    '''
    def get_layer_output(self, layer_names_list):
        return [self.layers[layer_name] for layer_name in layer_names_list]
