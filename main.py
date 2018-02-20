from ArtisticNet import ArtisticNet
import os
import sys
import argparse

def _get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("content_image", action='store', help="The path to the content image")
    parser.add_argument("style_image", action='store', help="The path to the style image")
    parser.add_argument("--output-name", dest="output_name", action='store', help="The output file name. Default: output")
    parser.add_argument('--image-height', dest='image_height', action='store',
                      help='The height of the output image. Default: The content image height', type=int)
    parser.add_argument('--image-width', dest='image_width', action='store',
                      help='The width of the output image. Default: The content image width', type=int)
    parser.add_argument('--optimizer_method', dest='optimizer_method', action='store',
                      choices=['L-BFGS','Adam'],
                      help='Optimizer method. Default: L-BFGS')
    parser.add_argument('--model', dest='model', action='store',
                      choices=['VGG19','GOOGLENET'],
                      help='The CNN model to use. Default: VGG19. If using GoogLeNet, you should specify content and style feature layers')
    parser.add_argument('--alpha', dest='alpha', action='store',
                      help='The content weight in the total loss. Default: 2', type=float)
    parser.add_argument('--beta', dest='beta', action='store',
                      help='The style weight in the total loss. Default: 10', type=float)
    parser.add_argument('--num-steps', dest='num_steps', action='store',
                      help='The numbers of iteration to run. Default: 1000', type=int)
    parser.add_argument('--learning-rate', dest='learning_rate', action='store',
                      help='The learning rate. Default: 2', type=float)
    parser.add_argument('--device', dest='device', action='store',
                      choices=['/cpu:0','/gpu:0'],
                      help='The device to use. Default: /cpu:0')
    parser.add_argument('--content-feature-layer', dest='content_feature_layers', action='store',
                      help='The name of the content feature layer in the model. Default: conv4_2 for VGG19')
    parser.add_argument('--style-feature-layers', dest='style_feature_layers', nargs='+', action='store',
                      help='The name of the style feature layers in the model. Default: ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]')
    parser.add_argument('--weights-file-path', dest='mat_path', action='store',
                      help='The path to the weight mat file. Default: mat_files/imagenet-vgg-verydeep-19.mat')                  
    parser.add_argument('--content-loss-normalization', dest='content_loss_norm', action='store',
                      choices=['0','1','2'],
                      help='The normalization for the content loss: 0 - factor 0.5, 1 - factor of 1/(4*N*M), 2 - 1/(4*sqrt(N)*sqrt(M). Default: 2', type=int)
    parser.add_argument('--init-image-method', dest='noise_img_method', action='store',
                      choices=['truncated_normal','content_variance','noisy_content','content','style','style_variance'],
                      help='The method for generating the initial image. Default: content_variance')
    parser.add_argument('--noise-ratio', dest='noise_ratio', action='store',
                      help='The percentage of noise when using white-noise initial image method. Default: 0.6', type=float)
    parser.add_argument('--noise-mean', dest='mean', action='store',
                      help='The mean of the white-noise. Default: 0', type=float)
    parser.add_argument('--noise-variance', dest='variance', action='store',
                      help='The variance of the white-noise. Default: 1', type=float)
    parser.add_argument('--variance-ratio', dest='variance_ratio', action='store',
                      help='The scaling to the variance when using white-noise initial image method. Default: 0.1', type=float)
    parser.add_argument('--style-weights-list', dest='w_l_list', nargs='+', action='store',
                      help='The weights of the style feature layers. Default: [0.2, 0.2, 0.2, 0.2, 0.2]', type=float)
                      
    return parser
 
if __name__ == "__main__":
    args_parser = _get_args_parser()
    args = args_parser.parse_args(sys.argv[1:])
    
    args = vars(args)
    
   # Remove None values that may be found in args due to the use of argparse
    args = dict((k, v) for k, v in args.items() if v)

    artisticNet = ArtisticNet(**args)
    artisticNet.activate_network()
