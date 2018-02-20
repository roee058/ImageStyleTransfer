import tensorflow as tf
import numpy as np
import scipy.io
from CNN import CNN

# Values are taken from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
rgb_means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


class GOOGLENET_CNN(CNN):

    '''
    Initialize Googlenet network with parameters from .mat file
    @param param_mat_file_path - Full path of the .mat file
    @param image_height
    @param image_width
    '''
    def __init__(self, param_mat_file_path, image_height, image_width):
        super(GOOGLENET_CNN, self).__init__()
        self.params_dict = scipy.io.loadmat(param_mat_file_path)
        self.build_network(image_height, image_width)


    '''
    Will perform full activation of the network and will return the matching output by the input layers names list.
    @param image:             RGB image [batch, IMAGE_SIZE, IMAGE_SIZE, 3]
    @param layers_names_list: List of the requested layers output
    '''
    def activate_network(self, image, layers_names_list):
        img = image - rgb_means
        return super(GOOGLENET_CNN, self).activate_network(img, layers_names_list)

    '''
    Build the GOOGLENET network - the original GOOGLENET without the dropout and linear layers
    '''
    def build_network(self, image_height, image_width):
        self.network_input = tf.Variable(tf.zeros((1,image_height, image_width, 3)), dtype='float32')
        

        self.layers = {}
        
        # conv1
        self.layers['conv1']            = self.conv_layer(self.network_input, 2, 2, "conv1")
        self.layers['pool1']            = self.avg_pool(self.layers['conv1'], 3, 3, 2, 2, "pool1")
        
        # conv2
        self.layers['reduction2']       = self.conv_layer(self.layers['pool1'], 1, 1, "reduction2")
        self.layers['conv2']            = self.conv_layer(self.layers['reduction2'], 1, 1, "conv2")
        self.layers['pool2']            = self.avg_pool(self.layers['conv2'], 3, 3, 2, 2, "pool2")

        # Inception1
        self.layers['icp1_reduction1']  = self.conv_layer(self.layers['pool2'], 1, 1, "icp1_reduction1")
        self.layers['icp1_reduction2']  = self.conv_layer(self.layers['pool2'], 1, 1, "icp1_reduction2")
        self.layers['pool3']            = self.avg_pool(self.layers['pool2'], 3, 3, 1, 1, "pool3")
        self.layers['icp1_out0']        = self.conv_layer(self.layers['pool2'], 1, 1, "icp1_out0")
        self.layers['icp1_out1']        = self.conv_layer(self.layers['icp1_reduction1'], 1, 1, "icp1_out1")
        self.layers['icp1_out2']        = self.conv_layer(self.layers['icp1_reduction2'], 1, 1, "icp1_out2")
        self.layers['icp1_out3']        = self.conv_layer(self.layers['pool3'], 1, 1, "icp1_out3")
        self.layers['icp1_out']         = tf.concat(3, [self.layers['icp1_out0'], self.layers['icp1_out1'], self.layers['icp1_out2'], self.layers['icp1_out3']], 'icp1_out')
        
        # Inception2
        self.layers['icp2_reduction1']  = self.conv_layer(self.layers['icp1_out'], 1, 1, "icp2_reduction1")
        self.layers['icp2_reduction2']  = self.conv_layer(self.layers['icp1_out'], 1, 1, "icp2_reduction2")
        self.layers['pool4']            = self.avg_pool(self.layers['icp1_out'], 3, 3, 1, 1, "pool4")
        self.layers['icp2_out0']        = self.conv_layer(self.layers['icp1_out'], 1, 1, "icp2_out0")
        self.layers['icp2_out1']        = self.conv_layer(self.layers['icp2_reduction1'], 1, 1, "icp2_out1")
        self.layers['icp2_out2']        = self.conv_layer(self.layers['icp2_reduction2'], 1, 1, "icp2_out2")
        self.layers['icp2_out3']        = self.conv_layer(self.layers['pool4'], 1, 1, "icp2_out3")
        self.layers['icp2_out']         = tf.concat(3, [self.layers['icp2_out0'], self.layers['icp2_out1'], self.layers['icp2_out2'], self.layers['icp2_out3']], 'icp2_out')

        # pool5
        self.layers['pool5']            = self.avg_pool(self.layers['icp2_out'], 3, 3, 2, 2, "pool5")

        # Inception3
        self.layers['icp3_reduction1']  = self.conv_layer(self.layers['pool5'], 1, 1, "icp3_reduction1")
        self.layers['icp3_reduction2']  = self.conv_layer(self.layers['pool5'], 1, 1, "icp3_reduction2")
        self.layers['pool6']            = self.avg_pool(self.layers['pool5'], 3, 3, 1, 1, "pool6")
        self.layers['icp3_out0']        = self.conv_layer(self.layers['pool5'], 1, 1, "icp3_out0")
        self.layers['icp3_out1']        = self.conv_layer(self.layers['icp3_reduction1'], 1, 1, "icp3_out1")
        self.layers['icp3_out2']        = self.conv_layer(self.layers['icp3_reduction2'], 1, 1, "icp3_out2")
        self.layers['icp3_out3']        = self.conv_layer(self.layers['pool6'], 1, 1, "icp3_out3")
        self.layers['icp3_out']         = tf.concat(3, [self.layers['icp3_out0'], self.layers['icp3_out1'], self.layers['icp3_out2'], self.layers['icp3_out3']], 'icp3_out')

        # Inception4
        self.layers['icp4_reduction1']  = self.conv_layer(self.layers['icp3_out'], 1, 1, "icp4_reduction1")
        self.layers['icp4_reduction2']  = self.conv_layer(self.layers['icp3_out'], 1, 1, "icp4_reduction2")
        self.layers['pool7']            = self.avg_pool(self.layers['icp3_out'], 3, 3, 1, 1, "pool7")
        self.layers['icp4_out0']        = self.conv_layer(self.layers['icp3_out'], 1, 1, "icp4_out0")
        self.layers['icp4_out1']        = self.conv_layer(self.layers['icp4_reduction1'], 1, 1, "icp4_out1")
        self.layers['icp4_out2']        = self.conv_layer(self.layers['icp4_reduction2'], 1, 1, "icp4_out2")
        self.layers['icp4_out3']        = self.conv_layer(self.layers['pool7'], 1, 1, "icp4_out3")
        self.layers['icp4_out']         = tf.concat(3, [self.layers['icp4_out0'], self.layers['icp4_out1'], self.layers['icp4_out2'], self.layers['icp4_out3']], 'icp4_out')

        # Inception5
        self.layers['icp5_reduction1']  = self.conv_layer(self.layers['icp4_out'], 1, 1, "icp5_reduction1")
        self.layers['icp5_reduction2']  = self.conv_layer(self.layers['icp4_out'], 1, 1, "icp5_reduction2")
        self.layers['pool8']            = self.avg_pool(self.layers['icp4_out'], 3, 3, 1, 1, "pool8")
        self.layers['icp5_out0']        = self.conv_layer(self.layers['icp4_out'], 1, 1, "icp5_out0")
        self.layers['icp5_out1']        = self.conv_layer(self.layers['icp5_reduction1'], 1, 1, "icp5_out1")
        self.layers['icp5_out2']        = self.conv_layer(self.layers['icp5_reduction2'], 1, 1, "icp5_out2")
        self.layers['icp5_out3']        = self.conv_layer(self.layers['pool8'], 1, 1, "icp5_out3")
        self.layers['icp5_out']         = tf.concat(3, [self.layers['icp5_out0'], self.layers['icp5_out1'], self.layers['icp5_out2'], self.layers['icp5_out3']], 'icp5_out')

        # Inception6
        self.layers['icp6_reduction1']  = self.conv_layer(self.layers['icp5_out'], 1, 1, "icp6_reduction1")
        self.layers['icp6_reduction2']  = self.conv_layer(self.layers['icp5_out'], 1, 1, "icp6_reduction2")
        self.layers['pool9']            = self.avg_pool(self.layers['icp5_out'], 3, 3, 1, 1, "pool9")
        self.layers['icp6_out0']        = self.conv_layer(self.layers['icp5_out'], 1, 1, "icp6_out0")
        self.layers['icp6_out1']        = self.conv_layer(self.layers['icp6_reduction1'], 1, 1, "icp6_out1")
        self.layers['icp6_out2']        = self.conv_layer(self.layers['icp6_reduction2'], 1, 1, "icp6_out2")
        self.layers['icp6_out3']        = self.conv_layer(self.layers['pool9'], 1, 1, "icp6_out3")
        self.layers['icp6_out']         = tf.concat(3, [self.layers['icp6_out0'], self.layers['icp6_out1'], self.layers['icp6_out2'], self.layers['icp6_out3']], 'icp6_out')
        
        # Inception7
        self.layers['icp7_reduction1']  = self.conv_layer(self.layers['icp6_out'], 1, 1, "icp7_reduction1")
        self.layers['icp7_reduction2']  = self.conv_layer(self.layers['icp6_out'], 1, 1, "icp7_reduction2")
        self.layers['pool10']            = self.avg_pool(self.layers['icp6_out'], 3, 3, 1, 1, "pool10")
        self.layers['icp7_out0']        = self.conv_layer(self.layers['icp6_out'], 1, 1, "icp7_out0")
        self.layers['icp7_out1']        = self.conv_layer(self.layers['icp7_reduction1'], 1, 1, "icp7_out1")
        self.layers['icp7_out2']        = self.conv_layer(self.layers['icp7_reduction2'], 1, 1, "icp7_out2")
        self.layers['icp7_out3']        = self.conv_layer(self.layers['pool10'], 1, 1, "icp7_out3")
        self.layers['icp7_out']         = tf.concat(3, [self.layers['icp7_out0'], self.layers['icp7_out1'], self.layers['icp7_out2'], self.layers['icp7_out3']], 'icp7_out')
        
        # pool11
        self.layers['pool11']            = self.avg_pool(self.layers['icp7_out'], 3, 3, 2, 2, "pool11")
        
        # Inception8
        self.layers['icp8_reduction1']  = self.conv_layer(self.layers['pool11'], 1, 1, "icp8_reduction1")
        self.layers['icp8_reduction2']  = self.conv_layer(self.layers['pool11'], 1, 1, "icp8_reduction2")
        self.layers['pool12']            = self.avg_pool(self.layers['pool11'], 3, 3, 1, 1, "pool12")
        self.layers['icp8_out0']        = self.conv_layer(self.layers['pool11'], 1, 1, "icp8_out0")
        self.layers['icp8_out1']        = self.conv_layer(self.layers['icp8_reduction1'], 1, 1, "icp8_out1")
        self.layers['icp8_out2']        = self.conv_layer(self.layers['icp8_reduction2'], 1, 1, "icp8_out2")
        self.layers['icp8_out3']        = self.conv_layer(self.layers['pool12'], 1, 1, "icp8_out3")
        self.layers['icp8_out']         = tf.concat(3, [self.layers['icp8_out0'], self.layers['icp8_out1'], self.layers['icp8_out2'], self.layers['icp8_out3']], 'icp8_out')
        
        # Inception9
        self.layers['icp9_reduction1']  = self.conv_layer(self.layers['icp8_out'], 1, 1 , "icp9_reduction1")
        self.layers['icp9_reduction2']  = self.conv_layer(self.layers['icp8_out'], 1, 1, "icp9_reduction2")
        self.layers['pool13']            = self.avg_pool(self.layers['icp8_out'], 3, 3, 1, 1, "pool13")
        self.layers['icp9_out0']        = self.conv_layer(self.layers['icp8_out'], 1, 1, "icp9_out0")
        self.layers['icp9_out1']        = self.conv_layer(self.layers['icp9_reduction1'], 1, 1, "icp9_out1")
        self.layers['icp9_out2']        = self.conv_layer(self.layers['icp9_reduction2'], 1, 1, "icp9_out2")
        self.layers['icp9_out3']        = self.conv_layer(self.layers['pool13'], 1, 1, "icp9_out3")
        self.layers['icp9_out']         = tf.concat(3, [self.layers['icp9_out0'], self.layers['icp9_out1'], self.layers['icp9_out2'], self.layers['icp9_out3']], 'icp9_out')
                
        # avg_pool
        self.layers['pool14']            = self.avg_pool(self.layers['icp9_out'], 7, 7, 1, 1, "pool14")

        
        self.last_layer = self.layers['pool14']

    def max_pool(self, prev_layer, kernel_h, kernel_w, stride_h, stride_w, name):
        return tf.nn.max_pool(prev_layer, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1], padding='SAME', name=name)

    def avg_pool(self, prev_layer, kernel_h, kernel_w, stride_h, stride_w, name):
        return tf.nn.avg_pool(prev_layer, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1], padding='SAME', name=name)
        
        
    # Build convolution layer with weights from Mat file and RELU non-linear operation
    def conv_layer(self, prev_layer, stride_h, stride_w, name):
        with tf.variable_scope(name):
            weights = self.get_weights("{}_filter".format(name))
            conv = tf.nn.conv2d(prev_layer, weights, [1, stride_h, stride_w, 1], padding='SAME')
            conv_biases = self.get_bias("{}_bias".format(name))
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    # Return the weights from the parameters DB by their layer name
    def get_weights(self, name):
        layer_idx = self._get_layer_idx_by_name(name)
        weights = tf.constant(self.params_dict['params'][0][layer_idx][1])
        return weights

    # Return the weights from the parameters DB by their layer name
    def get_bias(self, name):
        layer_idx = self._get_layer_idx_by_name(name)
        bias = self.params_dict['params'][0][layer_idx][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return bias

    # Will return the matching index of the given layer name
    def _get_layer_idx_by_name(self, name):
        layer_num = self.params_dict['params'][0].shape[0]
        return [self.params_dict['params'][0][i][0][0] for i in range(0, layer_num)].index(name)

    # Print image to the requested path
    def print_image(self, img, path):
        img         = img + rgb_means
        img_clip    =  np.clip(img[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, img_clip)

