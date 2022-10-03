import SimpleITK as itk
import numpy as np
import os
from Autoinpainting.libs.img_utils import load_img_array2
from Autoinpainting.libs.paths_dirs_stuff import get_data_list, get_sub_dirs, creat_dir
from Autoinpainting.libs.sitk_stuff import read_nifti, get_axis_order_to_RAI


def image_to_nifti(n_subject_nifti, n_subject_input, n_subject_output,
                   nifti_subfolders, input_img_subfolder, output_img_subfolder,
                   ct_pattern, write_path_ct_res, write_path_pet_res):
    '''

    Load the subjectwise inpainted 2 images
    calculate the differences between original and inpainted images
    convert the residual CT and PET volumes into separate nifti files.
    ----------
    n_subject_nifti : int
        number of nifti subfolders.
    n_subject_input : int
        number of subfolders of original 2d images .
    n_subject_output : int
        number of subfolders of inpainted images.
    nifti_subfolders : list
        containing subfolders of nifti volumes.
    input_img_subfolder : list
        containing full dir to subfolders of original 2d images.
    output_img_subfolder : list
        containing full dir to subfolders of inapinted images.
    ct_pattern : str
        filename patterns of CT volumes.
    write_path_ct_res : str
        dir to save residual CT nifti.
    write_path_pet_res : str
        dir to save residual PET nifti.

    '''

    if n_subject_nifti != n_subject_input or n_subject_nifti != n_subject_output or n_subject_input != n_subject_output:
        raise Exception("number of image subjects are not matched btw input, output and niftis!")
    else:
        pass
    for ix in range(n_subject_input):
        print('Writing nifti files of subject {} out of {}'.format(ix, n_subject_input))

        nifti_folder = nifti_subfolders[ix]
        input_folder = input_img_subfolder[ix]
        output_folder = output_img_subfolder[ix]

        nifti_content = get_data_list(nifti_folder, ct_pattern)
        input_content = get_data_list(input_folder, '.png')
        output_content = get_data_list(output_folder, '.png')

        _, _, img_size, img_spacing, img_origin, img_direction, ct_orientation_matrix = read_nifti(nifti_content[0])
        nifti_width = img_size[0]
        nifti_height = img_size[1]
        nifti_depth = img_size[2]

        input_array = np.array(load_img_array2(input_content, nifti_width, nifti_height))
        inpaint_array = np.array(load_img_array2(output_content, nifti_width, nifti_height))

        resiual_array = input_array - inpaint_array
        resiual_array_ct = resiual_array[:, :, :, 0]
        residual_array_pet = resiual_array[:, :, :, 1]

        ct_axis_orientation, ct_flip_axes = get_axis_order_to_RAI(ct_orientation_matrix)

        ORIENTATION_MAP = {"sagittal": 2, "coronal": 1, "axial": 0}

        orientation_index = ORIENTATION_MAP['axial']
        axis_index = ct_axis_orientation.index(orientation_index)
        axis_to_flip = [axis for axis, flip in enumerate(ct_flip_axes) if flip]

        resiual_array_ct = np.swapaxes(resiual_array_ct, 0, axis_index)
        resiual_array_ct = np.flip(resiual_array_ct, axis_to_flip)

        residual_array_pet = np.swapaxes(residual_array_pet, 0, axis_index)
        residual_array_pet = np.flip(residual_array_pet, axis_to_flip)

        if resiual_array_ct.shape[0] != residual_array_pet.shape[0]:
            raise Exception('number of input and inpainted slices of subject {} are not matched'.format(input_folder))
        elif resiual_array_ct.shape[0] != nifti_depth:
            raise Exception('number of 2d and nifti slices of subject {} are not matched'.format(nifti_folder))
        else:
            pass
        res_ct_itk = itk.GetImageFromArray(resiual_array_ct)
        res_pet_itk = itk.GetImageFromArray(residual_array_pet)

        res_ct_itk.SetOrigin(img_origin)
        res_pet_itk.SetOrigin(img_origin)

        res_ct_itk.SetSpacing(img_spacing)
        res_pet_itk.SetSpacing(img_spacing)

        res_ct_itk.SetDirection(img_direction)
        res_pet_itk.SetDirection(img_direction)

        subject_name = nifti_folder.split('/')[-1]
        subject_name_ct = subject_name + '_ct_res.nii.gz'
        subject_name_pet = subject_name + '_pet_res.nii.gz'

        ct_res_path = os.path.join(write_path_ct_res, subject_name, subject_name_ct)
        pet_res_path = os.path.join(write_path_pet_res, subject_name, subject_name_pet)

        itk.WriteImage(res_ct_itk, ct_res_path)
        itk.WriteImage(res_pet_itk, pet_res_path)

    return None
