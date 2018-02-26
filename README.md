A TensorFlow implementation of Gatys et al Image Style Transfer using Convolutional Neural Networks

Given a content image and a style image, the goal is to generate a new image that preserves the main features of the content image but contains the texture of the style image. 

usage: main.py [-h] [--output-name OUTPUT_NAME] [--image-height IMAGE_HEIGHT]
               [--image-width IMAGE_WIDTH] [--optimizer_method {L-BFGS,Adam}]
               [--model {VGG19,GOOGLENET}] [--alpha ALPHA] [--beta BETA]
               [--num-steps NUM_STEPS] [--learning-rate LEARNING_RATE]
               [--device {/cpu:0,/gpu:0}]
               [--content-feature-layer CONTENT_FEATURE_LAYERS]
               [--style-feature-layers STYLE_FEATURE_LAYERS [STYLE_FEATURE_LAYERS ...]]
               [--weights-file-path MAT_PATH]
               [--content-loss-normalization {0,1,2}]
               [--init-image-method {truncated_normal,content_variance,noisy_content,content,style,style_variance}]
               [--noise-ratio NOISE_RATIO] [--noise-mean MEAN]
               [--noise-variance VARIANCE] [--variance-ratio VARIANCE_RATIO]
               [--style-weights-list W_L_LIST [W_L_LIST ...]]
               content_image style_image

positional arguments:
  content_image         The path to the content image
  style_image           The path to the style image

optional arguments:
  -h, --help            show this help message and exit
  --output-name OUTPUT_NAME
                        The output file name. Default: output
  --image-height IMAGE_HEIGHT
                        The height of the output image. Default: The content
                        image height
  --image-width IMAGE_WIDTH
                        The width of the output image. Default: The content
                        image width
  --optimizer_method {L-BFGS,Adam}
                        Optimizer method. Default: L-BFGS
  --model {VGG19,GOOGLENET}
                        The CNN model to use. Default: VGG19. If using
                        GoogLeNet, you should specify content and style
                        feature layers
  --alpha ALPHA         The content weight in the total loss. Default: 2
  --beta BETA           The style weight in the total loss. Default: 10
  --num-steps NUM_STEPS
                        The numbers of iteration to run. Default: 1000
  --learning-rate LEARNING_RATE
                        The learning rate. Default: 2
  --device {/cpu:0,/gpu:0}
                        The device to use. Default: /cpu:0
  --content-feature-layer CONTENT_FEATURE_LAYERS
                        The name of the content feature layer in the model.
                        Default: conv4_2 for VGG19
  --style-feature-layers STYLE_FEATURE_LAYERS [STYLE_FEATURE_LAYERS ...]
                        The name of the style feature layers in the model.
                        Default: ["conv1_1", "conv2_1", "conv3_1", "conv4_1",
                        "conv5_1"]
  --weights-file-path MAT_PATH
                        The path to the weight mat file. Default:
                        mat_files/imagenet-vgg-verydeep-19.mat
  --content-loss-normalization {0,1,2}
                        The normalization for the content loss: 0 - factor
                        0.5, 1 - factor of 1/(4*N*M), 2 -
                        1/(4*sqrt(N)*sqrt(M). Default: 2
  --init-image-method {truncated_normal,content_variance,noisy_content,content,style,style_variance}
                        The method for generating the initial image. Default:
                        content_variance
  --noise-ratio NOISE_RATIO
                        The percentage of noise when using white-noise initial
                        image method. Default: 0.6
  --noise-mean MEAN     The mean of the white-noise. Default: 0
  --noise-variance VARIANCE
                        The variance of the white-noise. Default: 1
  --variance-ratio VARIANCE_RATIO
                        The scaling to the variance when using white-noise
                        initial image method. Default: 0.1
  --style-weights-list W_L_LIST [W_L_LIST ...]
                        The weights of the style feature layers. Default:
                        [0.2, 0.2, 0.2, 0.2, 0.2]

                        
Please download the MAT files from: https://drive.google.com/drive/folders/0Bzo638c-lYwaUmkwaV9lTlVqOG8
Please insert them in the mat_files directory
                        
 For example, to generate the output from content of Tel-Aviv and style of Starry-night, please use the following command for the following networks:
 
 VGG19:
    python main.py ../content_images/tlv.jpg ../style_images/starry_night.jpg --image-height 400 --image-width 600 --device /gpu:0
GOOGLENET:
    python main.py ../content_images/tlv.jpg ../style_images/starry_night.jpg --image-height 400 --image-width 600 --device /gpu:0 --weights-file-path ../mat_files/imagenet-googlenet-dag.mat --model GOOGLENET --style-feature-layers "conv1" "conv2" "icp1_out" "icp2_out" "icp3_out" --content-feature-layer pool2 --alpha 1e-1 --beta 10 --style-weights-list 0.2 0.2 0.2 0.2 0.2
