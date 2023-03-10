import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def create_gaussian_filter(ksize, sigma):
    assert ksize % 2 == 1

    import numpy as np
    oneD = np.arange(-(ksize // 2), (ksize // 2) + 1, 1)

    x, y = np.meshgrid(oneD, oneD)

    matrix = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return matrix


def pad_image_reflection(img, x2, y2):
    # Input: Image to be padded, x2,y2 which are size of kernel applied
    # e.g if you convolve image with 9*9 kernel then you need to pad
    # 8 pixels from each side then x2=9, y2=9
    x1, y1 = img.shape

    # Bonus
    left = np.fliplr(img[:, 0:x2 // 2])
    right = (img[:, -1:-1 - (x2 // 2):-1])
    horiz_padded = np.hstack((left, img, right))

    up = np.flipud(horiz_padded[0:y2 // 2, :])
    down = (horiz_padded[-1:-1 - (y2 // 2):-1, :])
    output = np.vstack((up, horiz_padded, down))

    return output


def conv_2d(img, kernel):
    # This function gets 2 matrices as input
    # and performs correlation between one them (kernel)
    # with all the pixels of the others
    # output is 1 2-D matrix of size = shape1 + shape2 -1
    import numpy as np
    x1, y1 = img.shape
    x2, y2 = kernel.shape

    # Zero padding for image
    zero_padded = pad_image_reflection(img, x2, y2)

    # Define output matrix
    output = np.zeros((x1, y1))
    # Multiply and add (correlate)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            sliced = zero_padded[i:i + x2, j:j + y2]
            output[i, j] = np.sum(sliced * kernel)

    return output


def conv_fft(img, kernel):
    from numpy.fft import fft2, ifft2
    height = max([img.shape[0], kernel.shape[0]])
    width = max([img.shape[1], kernel.shape[1]])

    # Get Freq domain of both images, multiply, then ifft
    img_fft = fft2(img, (height, width), (0, 1))
    kernel_fft = fft2(kernel, (height, width), (0, 1))
    output = np.abs(ifft2(img_fft * kernel_fft))

    # This part is for correction of effect of numpy padding
    # which reverse the order of some columns and rows
    h = min([img.shape[0], kernel.shape[0]])
    w = min([img.shape[1], kernel.shape[1]])
    output = np.hstack((output[:, w // 2:], output[:, 0:w // 2]))
    output = np.vstack((output[h // 2:, :], output[0:h // 2, :]))

    return output


def my_imfilter(img, kernel, type=1):
    output = np.zeros(img.shape)
    if img.ndim == 3:
        for channel in range(img.shape[2]):
            if type == 0:
                output[:, :, channel] = conv_2d(img[:, :, channel], kernel)
            else:
                output[:, :, channel] = conv_fft(img[:, :, channel], kernel)
    elif img.ndim == 2:
        if type == 0:
            output = conv_2d(img, kernel)
        else:
            output = conv_fft(img, kernel)

    return output


def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float,
                     k: float, type: bool):
    """
    Inputs:
    - image1 -> The image from which to take the low frequencies.
    - image2 -> The image from which to take the high frequencies.
    - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                          blur that will remove high frequencies.
    image is between 0 and 1 

    Task:
    - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
    - Combine them to create 'hybrid_image'.
    """

    assert image1.shape == image2.shape
    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
    # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
    # kernel = np.array([[0.1104,    0.1115,    0.1104,],
    # [0.1115,    0.1126,    0.1115],
    # [0.1104,    0.1115,    0.1104]])

    kernel = create_gaussian_filter(k, sigma=cutoff_frequency)
    # Your code here:
    low_frequencies = my_imfilter(image1, kernel, type)
    print("low_frequencies", low_frequencies.shape, (low_frequencies > 0.1).any())
    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    unity = np.zeros((k, k))
    unity[k // 2, k // 2] = 1
    high_frequencies = my_imfilter(image2, unity, type) - my_imfilter(image2, kernel,
                                                                      type)  # Replace with your implementation
    # high_frequencies+=0.5
    print("high_frequencies", high_frequencies.shape, (high_frequencies > 0.1).any())

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies  # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    low_frequencies, high_frequencies, hybrid_image = low_frequencies.clip(0, 1), high_frequencies.clip(0,
                                                                                                        1), hybrid_image.clip(
        0, 1)
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.
    # (5) As a good software development practice you may add some checks (assertions) for the shapes
    # and ranges of your results. This can be performed as test for the code during development or even
    # at production!
    assert hybrid_image.min() >= 0 and hybrid_image.max() <= 1
    # raise("")
    return low_frequencies, high_frequencies, hybrid_image


def vis_hybrid_image(hybrid_image: np.ndarray):
    """
    Visualize a hybrid image by progressively downsampling the image and
    concatenating all of the images together.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect', channel_axis=2)
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    """Convert an image to single-precision (32-bit) floating point format, with values in [0, 1].
    """
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))
