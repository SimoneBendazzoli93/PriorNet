import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from Autoinpainting.libs.anomaly_tools import get_candid_mask, get_pred_candid
from Autoinpainting.libs.anomaly_tools import get_circle_candids, get_pred_random_masks
from Autoinpainting.libs.anomaly_tools import get_organ_mask, gen_random_masks_in_lung
from Autoinpainting.libs.img_utils import load_img_array
from Autoinpainting.libs.paths_dirs_stuff import get_data_list, creat_dir
from scipy.ndimage import morphology


def self_inpaint(img_subjects, map_subjects, n_subjects, output_dir,
                 image_size, crop_vol_ind_buttom, crop_vol_ind_top,
                 n_top_inpaint, rad, interval_idx, n_top_circle_select,
                 model_multi, res_thr, checkpoint_dir):
    '''

    Read the subjectwise 2D images, inpaint the auto-candidate regions
    and save the inpainting results as 2D images.
    ----------
    img_subjects : list
        containing full dir to subjetwise 2D folders of image.
    map_subjects : list
        containing full dir to subjetwise 2D folders of binary maps.
    n_subjects : int
        number of folders(subjects).
    output_dir : str
        directory for saving the results.
    image_size : int
        size of 2d images (256 default for pretarined network).
    crop_vol_ind_buttom : int
        lower boundary of volume slices to be processed (skip legs).
    crop_vol_ind_top : int
        higher boundary of volume slices to be processed (skin brain).
    n_top_inpaint : int
        inpainting the top candidates.
    rad : int
        radius of moving circles.
    interval_idx : int
        distance between the consecutive moving circles.
    n_top_circle_select : int
        top candidate to be used for inpaiting.
    model_multi : 
        Compiled inpainting model with pretrained weights.
    res_thr : float number
        based on circle size normalization .
    checkpoint_dir : str
        dir to the model weights.

    '''

    elapse_time = []
    if len(img_subjects) != len(map_subjects):
        raise Exception("number of image subject are not matched with number of map subjects!")
    else:
        pass
    for ind in range(n_subjects):
        print('Inpainting subject {} out of {}'.format(ind, n_subjects))
        subject_img_path = img_subjects[ind]
        subject_map_path = map_subjects[ind]

        subject_name_img = subject_img_path.split('/')[-1]
        subject_name_map = subject_map_path.split('/')[-1]
        if subject_name_img != subject_name_map:
            raise Exception("subject image and subject map are not matched!")
        else:
            pass
        subject_write_dir = os.path.join(output_dir, subject_name_img)
        creat_dir(subject_write_dir)

        img_lists = get_data_list(subject_img_path, '.png')
        map_lists = get_data_list(subject_map_path, '.png')

        img_array = load_img_array(img_lists, image_size)
        map_array = load_img_array(map_lists, image_size)
        map_array[map_array > 0.5] = 1
        organ_mask_volume = get_organ_mask(img_array)

        # slice_n_base = img_lists[0].split('.png')[0].split('_')[-1]
        # slice_n_base = int(slice_n_base)+1

        n_slice = img_array.shape[0]

        initTime = time.time()
        for ix in range(n_slice):

            slice_name = subject_name_img + '_inpaint_' + str(ix) + '.png'
            path_to_slice = os.path.join(subject_write_dir, slice_name)
            main_img = img_array[ix]
            pet_slice = main_img[:, :, 1]
            pet_slice = np.repeat(pet_slice[:, :, np.newaxis], 3, axis=2).astype('float32')

            if ix >= crop_vol_ind_buttom and ix < n_slice + crop_vol_ind_top:
                main_field_map = map_array[ix]
                main_field_map = main_field_map[:, :, 0].astype('int')
                main_field_map = morphology.binary_dilation(main_field_map, structure=np.ones((13, 13)))
                main_field_map = np.repeat(main_field_map[:, :, np.newaxis], 3, axis=2).astype('float32')
                organ_mask = organ_mask_volume[ix]

                n_top = n_top_inpaint
                random_masks = gen_random_masks_in_lung(main_field_map, main_img, radius=rad, interval_idx=interval_idx)
                if len(random_masks) < n_top_circle_select:
                    n_top_circle_select = len(random_masks)
                selects_ind = get_circle_candids(random_masks, pet_slice, n_top_circle_select)
                random_masks_selects = [random_masks[ii] for ii in selects_ind]
                n_random_mask_select = len(random_masks_selects)
                field_sum_int = np.sum(main_field_map)

                if field_sum_int > 40:  # instead of 0 avoid tiny structures
                    if n_random_mask_select > 0:
                        if n_top >= n_random_mask_select:
                            n_top = n_random_mask_select
                        else:
                            pass

                        pred_imgs, corrupt_imgs, resid_ints = get_pred_random_masks(random_masks_selects, main_img,
                                                                                    model_multi, organ_mask)
                        union_mask = get_candid_mask(resid_ints, random_masks_selects, rad, n_top, res_thr)
                        union_mask = union_mask * organ_mask

                        corrupt_img, my_pred = get_pred_candid(main_img, union_mask, model_multi)
                        # tmp_test = np.concatenate((main_img, corrupt_img, my_pred,corrupt_imgs[4]), axis=1)
                        # plt.imshow(tmp_test)
                        cv2.imwrite(path_to_slice, my_pred * 255)
                    else:
                        cv2.imwrite(path_to_slice, main_img * 255)
                else:
                    cv2.imwrite(path_to_slice, main_img * 255)
            else:
                cv2.imwrite(path_to_slice, main_img * 255)

        FinalTime = time.time() - initTime
        elapse_time.append(FinalTime)

    params = {}
    params['circle_rad'] = rad
    params['image_size'] = image_size
    params['circle_area_thr'] = res_thr
    params['interval_idx'] = interval_idx
    params['n_top_candidate'] = n_top_inpaint
    params['model_weight_path'] = checkpoint_dir
    params['crop_vol_ind_top'] = crop_vol_ind_top
    params['elapsed_time_per_subject'] = elapse_time
    params['crop_vol_ind_buttom'] = crop_vol_ind_buttom
    params['n_top_circle_select'] = n_top_circle_select

    json_name = 'Config_Autoinapint.json'
    with open(os.path.join(output_dir, json_name), 'w') as fp:
        json.dump(params, fp, indent=4)

    return None
