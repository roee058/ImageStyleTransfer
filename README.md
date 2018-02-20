A TensorFlow implementation of Gatys et al Image Style Transfer using Convolutional Neural Networks

Given a content image and a style image, the goal is to generate a new image that preserves the main features of the content image but contains the texture of the style image. 

Current support of models - VGG19, GoogleNet (mat files with trained weights needs to be downloaded)

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
