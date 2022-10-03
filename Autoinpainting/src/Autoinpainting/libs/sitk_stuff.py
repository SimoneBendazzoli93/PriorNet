import SimpleITK as itk
import numpy as np


def get_axis_order_to_RAI(rotation_matrix):
    axis_orientation = list(np.argmax(np.abs(rotation_matrix), axis=1))

    flip_map = [False, False, False]

    oriented_rotation_matrix = []
    for idx in range(3):
        oriented_rotation_matrix.append(rotation_matrix[axis_orientation.index(idx)])

    axis = np.identity(3)
    for dim in range(3):
        projection = np.dot(axis[dim], oriented_rotation_matrix[dim])
        if projection < 0:
            flip_map[dim] = True

    flip_axes = []
    for dim in range(3):
        flip_axes.append(flip_map[axis_orientation[dim]])

    return axis_orientation, flip_axes


def read_nifti(image_path):
    """
    Parameters
    ----------
    image_path : Full directory to the image data.

    Returns
    -------
    img_itk : the itk image.
    img_size: the tensor size of the itk image.
    img_spacing : image spacing of the read itk image.
    img_origin : the origin coordinate of the read itk image

    """

    img_itk = itk.ReadImage(image_path)

    img_size = img_itk.GetSize()
    img_spacing = img_itk.GetSpacing()
    img_origin = img_itk.GetOrigin()
    img_direction = img_itk.GetDirection()
    img_array = itk.GetArrayFromImage(img_itk)
    direction = np.array(img_itk.GetDirection()).reshape(3, 3)

    spacing = np.asarray(img_itk.GetSpacing())
    origin = np.asarray(img_itk.GetOrigin())

    direction = np.asarray(direction)
    sr = min(max(direction.shape[0], 1), 3)
    affine: np.ndarray = np.eye(sr + 1)
    affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
    affine[:sr, -1] = origin[:sr]
    flip_diag = [[-1, 1], [-1, -1, 1], [-1, -1, 1, 1]][sr - 1]  # itk to nibabel affine
    affine = np.diag(flip_diag) @ affine

    orientation_matrix = np.eye(3)
    orientation_matrix[0] = affine[0][:-1] / -spacing[0]
    orientation_matrix[1] = affine[1][:-1] / -spacing[1]
    orientation_matrix[2] = affine[2][:-1] / spacing[2]

    return img_itk, img_array, img_size, img_spacing, img_origin, img_direction, orientation_matrix


def get_subject_sequence(img_itk, img_size, img_spacing, img_origin, mask_direction):
    """
    Get the sequence from 4D image images
    Parameters
    ----------
    img_itk : ITK image
        ITK image of the original 4D image.
    img_size : tuple/list
        Representing the size of the original 4D image.
    img_spacing : tuple/list
        Representing the voxel resolution and slice thickness.
    img_origin : tuple/list
        origin coordinates off the image space.

    Returns
    -------
    itk_sequences : list
        Each item of the list is one sequence of the 4D image in
        ITK image format with the same spacing and origin of the main
        image.

    """
    n_sequence = img_size[-1]
    img_array = itk.GetArrayFromImage(img_itk)
    itk_sequences = []
    for item in range(n_sequence):
        img_sequence = img_array[item]
        itk_img_sequence = itk.GetImageFromArray(img_sequence)
        itk_img_sequence.SetOrigin(img_origin[:3])
        itk_img_sequence.SetSpacing(img_spacing[:3])
        itk_img_sequence.SetDirection(mask_direction)
        itk_sequences.append(itk_img_sequence)

    return itk_sequences


def reorient_itk(itk_img):
    '''

    Parameters
    ----------
    itk_img : itk image
        takes the already loaded itk image
        reorient the image into LPS cosine matrix

    Returns
    -------
    reor_array : numpy array
        array of the re-oriented volume.

    '''

    orientation_filter = itk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation("LPS")
    reoriented = orientation_filter.Execute(itk_img)
    reor_array = itk.GetArrayFromImage(reoriented)

    return reor_array
