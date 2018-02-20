import skimage.transform
import scipy.misc
import numpy as np

# Load an image from given path
# If height and width are different from 0, use them for resizing
# The returned image will be with rank of 4 : [1,height, width, 3]
# Will return also the height and width of the image
def load_image(path, height=0, width=0):
    img = scipy.misc.imread(path)
    if height != 0 and width != 0:
        img = skimage.transform.resize(img, (height, width), preserve_range=True, clip=False)
        img = np.copy(img).astype('uint8')
    img = np.reshape(img, ((1,) + img.shape))
    return img, img.shape[1], img.shape[2]
