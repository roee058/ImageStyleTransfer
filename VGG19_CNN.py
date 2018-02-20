import tensorflow as tf
import numpy as np
import scipy.io
from CNN import CNN

# Values are taken from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
rgb_means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

class VGG19_CNN(CNN):

    '''
    Initialize VGG19 network with parameters from .mat file
    @param param_mat_file_path - Full path of the .mat file
    @param image_height
    @param image_width
    '''
    def __init__(self, param_mat_file_path, image_height, image_width):
        super(VGG19_CNN, self).__init__()
        self.params_dict = scipy.io.loadmat(param_mat_file_path)
        self.build_network(image_height, image_width)


    '''
    Will perform full activation of the network and will return the output of the layers nmaed in the layers names list.
    @param image:             RGB image [batch, IMAGE_SIZE, IMAGE_SIZE, 3] 
    @param layers_names_list: List of the requested layers output
    '''
    def activate_network(self, image, layers_names_list):
        img = image - rgb_means
        return super(VGG19_CNN, self).activate_network(img, layers_names_list)

    '''
    Build the VGG network - 16 layers of 19 layers - the original VGG-19 without the fully connected layers
    '''
    def build_network(self, image_height, image_width):
        self.network_input = tf.Variable(tf.zeros((1,image_height, image_width, 3)), dtype='float32')

        self.layers = {}

        self.layers['conv1_1'] = self.conv_layer(self.network_input, "conv1_1")
        self.layers['conv1_2'] = self.conv_layer(self.layers['conv1_1'], "conv1_2")
        self.layers['pool1']   = self.avg_pool(self.layers['conv1_2'], "pool1")

        self.layers['conv2_1'] = self.conv_layer(self.layers['pool1'], "conv2_1")
        self.layers['conv2_2'] = self.conv_layer(self.layers['conv2_1'], "conv2_2")
        self.layers['pool2']   = self.avg_pool(self.layers['conv2_2'], "pool2")

        self.layers['conv3_1'] = self.conv_layer(self.layers['pool2'], "conv3_1")
        self.layers['conv3_2'] = self.conv_layer(self.layers['conv3_1'], "conv3_2")
        self.layers['conv3_3'] = self.conv_layer(self.layers['conv3_2'], "conv3_3")
        self.layers['conv3_4'] = self.conv_layer(self.layers['conv3_3'], "conv3_4")
        self.layers['pool3']   = self.avg_pool(self.layers['conv3_4'], "pool3")

        self.layers['conv4_1'] = self.conv_layer(self.layers['pool3'], "conv4_1")
        self.layers['conv4_2'] = self.conv_layer(self.layers['conv4_1'], "conv4_2")
        self.layers['conv4_3'] = self.conv_layer(self.layers['conv4_2'], "conv4_3")
        self.layers['conv4_4'] = self.conv_layer(self.layers['conv4_3'], "conv4_4")
        self.layers['pool4']   = self.avg_pool(self.layers['conv4_4'], "pool4")

        self.layers['conv5_1'] = self.conv_layer(self.layers['pool4'], "conv5_1")
        self.layers['conv5_2'] = self.conv_layer(self.layers['conv5_1'], "conv5_2")
        self.layers['conv5_3'] = self.conv_layer(self.layers['conv5_2'], "conv5_3")
        self.layers['conv5_4'] = self.avg_pool(self.layers['conv5_3'], "conv5_4")

        self.last_layer = self.layers['conv5_4']

    def avg_pool(self, prev_layer, name):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, prev_layer, name):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
      
    # Build convolution layer with weights from Mat file and RELU non-linear operation
    def conv_layer(self, prev_layer, name):
        with tf.variable_scope(name):
            weights = self.get_weights(name)
            conv = tf.nn.conv2d(prev_layer, weights, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    # Return the weights from the parameters DB by their layer name
    def get_weights(self, name):
        layer_idx = self._get_layer_idx_by_name(name)
        weights = tf.constant(self.params_dict['layers'][0][layer_idx][0][0][2][0][0])
        return weights

    # Return the weights from the parameters DB by their layer name
    def get_bias(self, name):
        layer_idx = self._get_layer_idx_by_name(name)
        bias = self.params_dict['layers'][0][layer_idx][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return bias

    # Will return the matching index of the given layer name
    def _get_layer_idx_by_name(self, name):
        layer_num = self.params_dict['layers'][0].shape[0]
        return [self.params_dict['layers'][0][i][0][0][0][0] for i in range(0, layer_num)].index(name)

    # Print image to the requested path
    def print_image(self, img, path):
        img         = img + rgb_means
        img_clip    =  np.clip(img[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, img_clip)

