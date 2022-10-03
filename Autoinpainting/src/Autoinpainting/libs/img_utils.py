import cv2
import numpy as np
import os


def normalize(image, min_bound, max_bound):
    '''
    Parameters
    ----------
    image : numpy array
        loaded array of nifti volume.
    min_bound : int
        minimum intensity window of CT.
    max_bound : int
        maximum intensity window of CT.

    Returns
    -------
    image : numpy array
        rescaled [0 to 1] the image volume into the intensity window.

    '''
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image


def load_img_array(img_list, image_size=512):
    '''
    Parameters
    ----------
    img_list : list
        paths to image slicres.
    image_size : int
        represent the image size like 256 or 512.

    Returns
    -------
    img_array : array
        stack of slices.

    '''
    img_array = []
    for item in img_list:
        img_arr = cv2.imread(item)
        if img_arr.shape[0] != image_size:
            img_arr = cv2.resize(img_arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img_arr = img_arr / 255.
        img_arr = img_arr.astype(np.float32)
        img_array.append(img_arr)
    img_array = np.array(img_array)
    return img_array


def load_img_array2(img_list, image_size1=512, image_size2=512):
    '''
    Parameters
    ----------
    img_list : list
        paths to image slicres.
    image_size : int
        represent the image size like 256 or 512.

    Returns
    -------
    img_array : array
        stack of slices.

    '''
    img_array = []
    for item in img_list:
        img_arr = cv2.imread(item)
        if img_arr.shape[0] != image_size1 or img_arr.shape[1] != image_size2:
            img_arr = cv2.resize(img_arr, (image_size1, image_size2), interpolation=cv2.INTER_LINEAR)
        img_arr = img_arr / 255.
        img_arr = img_arr.astype(np.float32)
        img_array.append(img_arr)
    img_array = np.array(img_array)
    return img_array


def slice_writer(slice_array, dim1, dim2, slice_path):
    '''

    Parameters
    ----------
    slice_array : numpy array
        np attay of image slice to be written.
    dim1 : int
        image dimension (row).
    dim2 : int
        image dimension (col).
    slice_path : str
        fill directory including 'file_name.EXTENSION format'.

    Returns
    -------
    write 8bit image slices.

    '''

    img_slice = cv2.resize(slice_array, (dim1, dim2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(slice_path, img_slice * 255)

    return None


def write_multich(img, img_size, subject_write_dir, slice_name):
    img_slice = np.zeros((img_size, img_size, 3), 'float32')
    img_slice[:, :, 2] = img[:, :, 0]
    img_slice[:, :, 1] = img[:, :, 1]
    img_slice[:, :, 0] = img[:, :, 2]

    cv2.imwrite(os.path.join(subject_write_dir, slice_name), img_slice * 255)

    return None
