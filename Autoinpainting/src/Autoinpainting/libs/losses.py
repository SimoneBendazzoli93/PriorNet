import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def l1_error(y_true, y_pred):
    """Calculate the L1 loss used in all loss calculations"""
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3])
    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1, 2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")


def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""

    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"

    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]

    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H * W]))
    gram = K.batch_dot(features, features, axes=2)

    # Normalize with channels, height and width
    gram = gram / K.cast(C * H * W, x.dtype)

    return gram


def total_variation(y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""
    a = l1_error(y_comp[:, 1:, :, :], y_comp[:, :-1, :, :])
    b = l1_error(y_comp[:, :, 1:, :], y_comp[:, :, :-1, :])
    return a + b


def gauss_kernel(size=5, sigma=1.0):
    grid = np.int32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2).astype('float32')
    kernel /= np.sum(kernel)
    return kernel


def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),
                          [k_size, k_size, 1, 1])
    t_kernel3 = tf.concat([t_kernel] * t_input.get_shape()[3], axis=2)
    t_result = t_input
    for r in range(repeats):
        t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
                                          strides=[1, stride, stride, 1], padding='SAME')
    return t_result


def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=1.3)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = tf.nn.avg_pool(t_gauss, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    t_pyr.append(current)
    return t_pyr


def laploss(t_img1, t_img2):
    max_levels = 3
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = [tf.norm(a - b, ord=1) / tf.cast(tf.size(a), tf.float32) for a, b in zip(t_pyr1, t_pyr2)]
    t_loss = tf.reduce_sum(t_losses) * tf.cast(tf.shape(t_img1)[0], tf.float32)  # mehdi change type
    return t_loss
