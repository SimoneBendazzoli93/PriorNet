import argparse
import cv2
import json
import numpy as np
import os
from Autoinpainting.libs.img_utils import normalize
from Autoinpainting.libs.paths_dirs_stuff import get_sub_dirs, get_data_list, creat_dir
from Autoinpainting.libs.sitk_stuff import read_nifti, get_axis_order_to_RAI


def prepare_2d(nifti_subfolders, img_dir, map_dir, ct_pattern, pet_pattern,
               min_bound_ct, max_bound_ct, min_bound_pet, max_bound_pet,
               min_bound_map, max_bound_map, image_size, write_dir, nifti_root):
    '''
    Reading the main directory containing subfolders with PET and CT volumes.
    Converting the volumes into 2D slices (multichannel).
    Parameters
    ----------
    nifti_subfolders : list
        containing the full dir to all subject folders.
    img_dir : str
        dir to save the 2D images.
    map_dir : str
        dir to save the 2D binary maps obtained from pet images.
    ct_pattern : str
        showing the filename pattern of CT images.
    pet_pattern : str
        showing the filename pattern of PET images
    min_bound_ct : int
        lower bound of histogram windowing for CT.
    max_bound_ct : int
        higher bound of histogram windowing for CT..
    min_bound_pet : int
        lower bound of histogram windowing for PET.
    max_bound_pet : int
        higher bound of histogram windowing for PET.
    min_bound_map : int
        lower bound of histogram windowing for PET binary maps.
    max_bound_map : int
        higherr bound of histogram windowing for PET binary maps.
    image_size : int
        size of 2D images (256 default for pretrained models).
    write_dir : str
        dir to write the results, logs and 2D files.
    nifti_root : str
        dir to main directory of all subjects with subfolders containing .nii files.


    '''

    problematic_case = []
    for item in enumerate(nifti_subfolders):
        if item[0] % 5 == 0:
            print('Preparing 2D slices is in progress...')

        subject_path = item[1]

        subject_name = subject_path.split('/')[-1]
        subject_path_2d = os.path.join(img_dir, subject_name)
        subject_path_map = os.path.join(map_dir, subject_name)
        creat_dir(subject_path_2d)
        creat_dir(subject_path_map)

        nifti_ct_path = get_data_list(subject_path, ct_pattern)[0]
        nifti_pet_path = get_data_list(subject_path, pet_pattern)[0]

        try:
            ct_itk, ct_array, ct_size, _, _, _, ct_orientation_matrix = read_nifti(nifti_ct_path)
            pet_itk, pet_array, pet_size, _, _, _, pet_orientation_matrix = read_nifti(nifti_pet_path)

            ct_axis_orientation, ct_flip_axes = get_axis_order_to_RAI(ct_orientation_matrix)

            ORIENTATION_MAP = {"sagittal": 2, "coronal": 1, "axial": 0}

            orientation_index = ORIENTATION_MAP['axial']
            axis_index = ct_axis_orientation.index(orientation_index)
            axis_to_flip = [axis for axis, flip in enumerate(ct_flip_axes) if flip]

            ct_array = np.flip(ct_array, axis_to_flip)
            ct_array = np.swapaxes(ct_array, 0, axis_index)

            pet_array = np.flip(pet_array, axis_to_flip)
            pet_array = np.swapaxes(pet_array, 0, axis_index)

            if ct_size != pet_size:
                raise Exception("Size of CT and PET volumes are not matched!")
            else:
                ct_array_norm = normalize(ct_array, min_bound_ct, max_bound_ct)
                pet_array_norm = normalize(pet_array, min_bound_pet, max_bound_pet)
                pet_map = normalize(pet_array, min_bound_map, max_bound_map)
                depth, width, height = ct_array.shape
                for idx in range(depth):

                    ct_slice = ct_array_norm[idx]
                    pet_slice = pet_array_norm[idx]
                    map_slice = pet_map[idx]
                    if width != image_size:
                        ct_slice = cv2.resize(ct_slice, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                        pet_slice = cv2.resize(pet_slice, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                        # map_slice = cv2.resize(map_slice, (image_size,image_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        pass

                    img_slice = np.zeros((image_size, image_size, 3), 'float32')
                    img_slice[:, :, 0] = ct_slice
                    img_slice[:, :, 1] = pet_slice
                    img_slice[:, :, 2] = ct_slice

                    slice_name = subject_name + '_slice_' + str(idx) + '.png'
                    silce_path = os.path.join(subject_path_2d, slice_name)
                    map_path = os.path.join(subject_path_map, slice_name)
                    cv2.imwrite(silce_path, img_slice * 255)
                    cv2.imwrite(map_path, map_slice * 255)
        except:
            print('subjcet {} could not be loaded'.format(subject_name))
            problematic_case.append(item[1])

    params = {}
    params['write_dir'] = write_dir
    params['image_size'] = image_size
    params['nifti_root'] = nifti_root
    params['ct_pattern'] = pet_pattern
    params['pet_pattern'] = pet_pattern
    params['max_bound_ct'] = max_bound_ct
    params['min_bound_ct'] = min_bound_ct
    params['min_bound_pet'] = min_bound_pet
    params['max_bound_pet'] = max_bound_pet
    params['min_bound_map'] = min_bound_map
    params['max_bound_map'] = max_bound_map
    params['problematic_case'] = problematic_case

    json_name = '2D/input/Config_2DPreparation.json'
    with open(os.path.join(write_dir, json_name), 'w') as fp:
        json.dump(params, fp, indent=4)

    return None
