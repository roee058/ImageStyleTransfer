import tensorflow as tf
import numpy as np
import os
import time
import scipy.io
import utils
from VGG19_CNN import VGG19_CNN
from GOOGLENET_CNN import GOOGLENET_CNN

CONTENT_FEATURE_LAYERS  = "conv4_2"
STYLE_FEATURE_LAYERS    = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
MAT_PATH                = "../mat_files/imagenet-vgg-verydeep-19.mat"
W_L_LIST                = [0.2, 0.2, 0.2, 0.2, 0.2]
ALPHA                   = 2
BETA                    = 10
NUM_STEPS               = 1000
LEARNING_RATE           = 2
DEVICE                  = '/cpu:0'
OUTPUT_PATH             = 'output'

# Model:
# 1. VGG19
# 2. GOOGLENET
MODEL = "VGG19"

NOISE_RATIO             = 0.6
MEAN                    = 0
VARIANCE                = 1
VARIANCE_RATIO          = 0.1

# The optimizer method, can be:
# 1. L-BFGS
# 2. Adam
OPTIMIZER_METHOD        = 'L-BFGS'

# The options for the the initial noise image are:
# 1. truncated_normal - Use only truncated normal with given mean, variance and noise_ratio
# 2. content_variance - Use the content variance with given mean, multiplied by a variance ratio and a noise ratio
# 3. noisy_content    - Use the content image multiplied with noise ratio
# 4. content          - Use the content image as the initialized image.
# 5. style            - Use the style image as the initialized image.
# 6. style_variance   - Use the style variance with given mean, multiplied by a variance ratio and a noise ratio
NOISE_IMG_METHOD        = "content_variance"

# The normalization factor for the content loss:
# 0 - normalization factor of 0.5, as appears in the paper
# 1 - normalization factor of 1/(4*N*M)
# 2 - normalization factor of 1/(4*sqrt(N)*sqrt(M))
CONTENT_LOSS_NORM = 2

class ArtisticNet:
    '''
    '''
    def __init__(self, content_image ,style_image,  **kwargs):

        # Get all user arguments
        self._parse_args(**kwargs)
        self.content_feature_layers = [self.content_feature_layers]
        
        # Open the images
        self.content_image_data, self.image_height, self.image_width = utils.load_image(content_image, self.image_height, self.image_width)
        self.style_image_data, _ , _                                 = utils.load_image(style_image, self.image_height, self.image_width)

        # If user inserted a network, use it, otherwise use VGG19
        if self.model == 'VGG19':
            self.cnn = VGG19_CNN(self.mat_path, self.image_height, self.image_width)
        elif self.model == 'GOOGLENET':
            self.cnn = GOOGLENET_CNN(self.mat_path, self.image_height, self.image_width)

    def _parse_args(self, **kwargs):
        self.model                  = kwargs.get('model', MODEL)
        self.mat_path               = kwargs.get('mat_path', MAT_PATH)
        self.content_feature_layers = kwargs.get('content_feature_layers', CONTENT_FEATURE_LAYERS)
        self.style_feature_layers   = kwargs.get('style_feature_layers', STYLE_FEATURE_LAYERS)
        self.image_height           = kwargs.get('image_height', 0)
        self.image_width            = kwargs.get('image_width', 0)
        self.w_l_list               = kwargs.get('w_l_list', W_L_LIST)
        self.alpha                  = kwargs.get('alpha', ALPHA)
        self.beta                   = kwargs.get('beta', BETA)
        self.num_steps              = kwargs.get('num_steps', NUM_STEPS)
        self.noise_ratio            = kwargs.get('noise_ratio', NOISE_RATIO)
        self.mean                   = kwargs.get('mean',MEAN)
        self.variance               = kwargs.get('variance', VARIANCE)
        self.variance_ratio         = kwargs.get('variance_ratio', VARIANCE_RATIO)
        self.noise_img_method       = kwargs.get('noise_img_method', NOISE_IMG_METHOD)
        self.learning_rate          = kwargs.get('learning_rate', LEARNING_RATE)
        self.optimizer_method       = kwargs.get('optimizer_method', OPTIMIZER_METHOD)
        self.content_loss_norm      = kwargs.get('content_loss_norm', CONTENT_LOSS_NORM)
        self.device                 = kwargs.get('device', DEVICE)
        self.output_path            = kwargs.get('output_path', OUTPUT_PATH)

    '''
    Activate the network with the given initial image, content image and style image.
    Returns a new stylished image
    '''
    def activate_network(self):

        # Activate the CNN on the content image and get the output of CONTENT_FEATURE_LAYERS
        content_feature_layers = self.cnn.activate_network(self.content_image_data, self.content_feature_layers)

        # Activate the CNN on the style image and get the output of STYLE_FEATURE_LAYERS
        style_feature_layers   = self.cnn.activate_network(self.style_image_data, self.style_feature_layers)

        # Create the loss function

        # Reshape the numpy arrays for loss function
        style_gram_matrix_list = self._get_gram_matrix_list(style_feature_layers, False)

        # Calculate the total loss
        loss = self._calculate_total_loss(content_feature_layers, style_gram_matrix_list)

        # Create the optimizer, based on Adam optimizer algorithm
        optimizer = self._get_optimizer(loss)

        # Generate the initilized noised image
        network_input = self._gen_noised_image()

        # Optimize the input image
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        
        print("Staring!")
        start_time = time.time()
        with tf.device(self.device), tf.Session(config = config) as session:
            self._optimize_stylished_image(optimizer, network_input, session, loss)
        print("Finished! Synthesis took {} seconds".format(time.time() - start_time))

    '''
    Generate and return the Gram matrix of the given feature filters
    '''
    def _get_gram_matrix_list(self, feature_layers, is_tensor):
        feature_layers   = self._reshape_features(feature_layers, is_tensor)
        return [tf.matmul(feature_layer, feature_layer, transpose_a = True) for feature_layer in feature_layers]

    '''
    Return the optimizer by the optimizer_method
    '''
    def _get_optimizer(self, loss):
        if self.optimizer_method == 'L-BFGS':
            return tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': self.num_steps})
        elif self.optimizer_method == 'Adam':
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    '''
    Perform the optimization on the initial image by the optimizer and the loss
    '''
    def _optimize_stylished_image(self, optimizer, stylished_image, session, loss):
        session.run(tf.global_variables_initializer())
        session.run(self.cnn.network_input.assign(stylished_image))
        if self.optimizer_method == 'L-BFGS':
            optimizer.minimize(session)
            [current_loss, img] = session.run([loss, self.cnn.network_input])
        elif self.optimizer_method == 'Adam':
            for step in range(self.num_steps):
                [current_loss, img] = session.run([loss, self.cnn.network_input])
 
        path = "{}.png".format(self.output_path)
        self.cnn.print_image(img, path)


    '''
    Caclculate the style loss by the following formula
        G_ij_l     = sum_k(F_ik_l * F_jk_l)
        E_l        = 1/(4N_l^2*M_l^2) * sum_ij (G_ij_l - A_ij_l)^2
        style_loss = sum_l(w_l * E_l)
    '''
    def _calculate_style_loss(self, style_gram_matrix_list, input_feature_layer_names):
        input_feature_layers   = [self.cnn.layers[input_feature_layer_name] for input_feature_layer_name in input_feature_layer_names]
        input_gram_matrix_list = self._get_gram_matrix_list(input_feature_layers, True)

        style_loss_list    = [tf.reduce_sum(tf.pow(G_l-A_l,2)) for G_l, A_l in zip(input_gram_matrix_list, style_gram_matrix_list)]

        normalized_factors = [1.0/(4.0 * (F_l.get_shape().as_list()[1]**2) * (F_l.get_shape().as_list()[2]**2) * (F_l.get_shape().as_list()[3]**2)) for F_l in input_feature_layers]
        style_loss_list    = [tf.mul(tf.mul(style_loss, normalized_factor),w_l) for style_loss, normalized_factor, w_l in zip(style_loss_list, normalized_factors, self.w_l_list)]
        out = tf.add_n(style_loss_list)
        return out

    '''
    Calculate the content loss by the following formula:
        content_loss = 0.5 * sum_ij((F_l - P_l)^2)
    @param content_feature_layeri = P_l
    @param input_feature_layer    = F_l
    '''
    def _calculate_content_loss(self, content_feature_layer, input_feature_name):
        input_layer = self.cnn.layers[input_feature_name]
        out         = tf.reduce_sum(tf.pow((input_layer - content_feature_layer), 2)) * self._get_content_loss_norm(content_feature_layer)
        return out

    '''
    Return the content normalization by the method
    '''
    def _get_content_loss_norm(self, content_feature_layer):
        if self.content_loss_norm == 0:
            return 0.5
        elif self.content_loss_norm == 1:
            return 1.0 / (4.0 * content_feature_layer.shape[1] * content_feature_layer.shape[2] * content_feature_layer.shape[3])
        elif self.content_loss_norm == 2:
            return 1.0 / (4.0 * (content_feature_layer.shape[1] * content_feature_layer.shape[2] * content_feature_layer.shape[3])**0.5)

    '''
    Calculate the total loss by the following formula:
        total_loss = alpha * content_loss + beta * style_loss
    '''
    def _calculate_total_loss(self, content_feature_layers, style_gram_matrix_list):
        content_loss = self._calculate_content_loss(content_feature_layers[0], self.content_feature_layers[0])
        style_loss   = self._calculate_style_loss(style_gram_matrix_list, self.style_feature_layers)
        return tf.add(tf.mul(content_loss, self.alpha), tf.mul(style_loss, self.beta))

    '''
    Reshape feature layers by the following shape
        [batch, width, height, depth] = shape
        new_shape = [width * height, depth]
    The reshapre will be done on a list of features
    '''
    def _reshape_features(self, features, is_tensor = False):
        if is_tensor:
            shapes = [feature.get_shape().as_list() for feature in features]
            return [tf.reshape(feature, [shape[1] * shape[2], shape[3]]) for feature, shape in zip(features, shapes)]
        else:
            shapes = [feature.shape for feature in features]
            return [feature.reshape(shape[1] * shape[2], shape[3]) for feature, shape in zip(features, shapes)]

    '''
    Will generate noised image by the input arguments
    '''
    def _gen_noised_image(self):
        if self.noise_img_method == "content_variance":
            sigma = np.std(self.content_image_data) * self.variance_ratio
            return np.random.normal(self.mean, sigma, self.content_image_data.shape) * self.noise_ratio
        if self.noise_img_method == "style_variance":
            sigma = np.std(self.style_image_data) * self.variance_ratio
            return np.random.normal(self.mean, sigma, self.style_image_data.shape) * self.noise_ratio
        if self.noise_img_method == "truncated_normal":
            return np.random.normal(self.mean, self.variance, self.content_image_data.shape) * self.noise_ratio
        if self.noise_img_method == "noisy_content":
            sigma = np.std(self.content_image_data) * self.variance_ratio
            return self.content_image_data * (1-self.noise_ratio) + np.random.normal(self.mean, sigma, self.content_image_data.shape) * self.noise_ratio
        if self.noise_img_method == "content":
            return self.content_image_data
        if self.noise_img_method == "style":
            return self.style_image_data       
