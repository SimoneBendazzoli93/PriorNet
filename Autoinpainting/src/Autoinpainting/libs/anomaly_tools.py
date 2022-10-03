import cv2
import numpy as np
from copy import deepcopy


def get_organ_mask(img_array):
    binary_maks_organ = np.zeros_like(img_array)
    ct_ch = img_array[:, :, :, 0]
    ct_ch = np.repeat(ct_ch[:, :, :, np.newaxis], 3, axis=3).astype('float32')
    xx = np.where(ct_ch >= .05)
    binary_maks_organ[xx] = 1
    return binary_maks_organ


def gen_random_masks_in_lung(lung_mask, main_img, radius=25, interval_idx=19):
    '''
    Create circular masks inside the lung region
    Parameters
    ----------
    lung_mask : array
        binary array representing the lung region.
    main_img : array
        main image array.
    radius : int, optional
        radius of the circular random mask. The default is 25.
    interval_idx : int, optional
        the distance btw every two circular masks. The default is 19.
    Returns
    -------
    random_masks : list
        a list of random mask, each item is a binary image.
    '''

    ind_lung = np.argwhere(lung_mask == 1)
    interval_ind = np.arange(0, len(ind_lung), radius * interval_idx)
    interval_lung_ind = ind_lung[interval_ind]

    random_masks = []
    for x_ind, y_ind, _ in interval_lung_ind:
        temp_img = np.zeros_like(main_img, np.uint8)
        mask_circle = cv2.circle(temp_img, (y_ind, x_ind), radius, (1, 1, 1), -1)
        random_masks.append(1 - mask_circle)

    return random_masks


def get_stats(non_zero_array):
    max_val = np.max(non_zero_array)
    mean_val = np.mean(non_zero_array)
    std_val = np.std(non_zero_array)

    return max_val, mean_val, std_val


def get_circle_candids(random_masks, pet_slice, n_top_circle):
    mean_val_ = []
    max_val_ = []
    std_val_ = []
    circle_candid_id = []
    for inx in range(len(random_masks)):

        rand_circle = random_masks[inx]
        rand_circle = 1 - rand_circle

        img_circle = pet_slice * rand_circle
        non_zeros = img_circle[np.where(img_circle != 0)]
        if len(non_zeros) > 0:  # regions like inside the lungs contains zero intensity
            max_val, mean_val, std_val = get_stats(non_zeros)
        else:
            max_val, mean_val, std_val = [0, 0, 0]
        mean_val_.append(mean_val)
        max_val_.append(max_val)
        std_val_.append(std_val)

    mean_val_array = np.array(mean_val_)
    top_mean_val = np.argpartition(mean_val_array, -n_top_circle)[-n_top_circle:]
    max_val_array = np.array(max_val_)
    top_max_val = np.argpartition(max_val_array, -n_top_circle)[-n_top_circle:]

    for isx in top_mean_val:
        corres_max = max_val_[isx]
        # print(corres_max)
        if corres_max >= .8:
            circle_candid_id.append(isx)

    return circle_candid_id


def get_pred_random_masks(random_masks, main_img, model, organ_mask):
    '''
    Inpaint the image for each of the circular holes
    Parameters
    ----------
    random_masks : list
        all the binary images with circular holes.
    main_img : array
        main image to be corrupted with circular masks for inpainting
        and processing.
    model : TenFlow model
        The loaded model with the proper weights.
    organ_mask : array
        field mask; restrict the calculations within the organ field region.
    Returns
    -------
    pred_imgs : list
        inpainted images for each of the corrupted image.
    corrupt_imgs : list
        the corrupted images with circular holes.
    resid_ints : list
        intensity diff between the original image and inpained one within the
        circular holes.
    '''
    resid_ints = []
    corrupt_imgs = []
    pred_imgs = []
    # diff_imgs = []
    for ind in range(len(random_masks)):
        orig_img = deepcopy(main_img)
        corrupt_img = deepcopy(main_img)
        mask = random_masks[ind]
        corrupt_img[mask == 0] = 1

        corrupt_img = np.expand_dims(corrupt_img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        my_pred = model.predict([corrupt_img, mask])

        corrupt_img = np.squeeze(corrupt_img)
        my_pred = np.squeeze(my_pred)
        # xx = np.concatenate((orig_img, np.squeeze(mask), corrupt_img, my_pred), axis=1)
        # plt.imshow(xx)

        field_mask = organ_mask
        field_mask[field_mask > 0] = 1  # just to make sure the tumor is included in the lung mask
        my_pred = my_pred * field_mask
        orig_img = orig_img * field_mask
        res_img = orig_img - my_pred
        res_img = res_img * np.squeeze(1 - mask)
        res_img[res_img < 0] = 0

        sum_int = np.sum(res_img)  # calculate the diff only inside the mask area
        resid_ints.append(sum_int)
        pred_imgs.append(my_pred)
        corrupt_imgs.append(corrupt_img)
        # diff_imgs.append(res_img)

        # lung_mask = lung_mask[:,:,0] # make it work in the loop

    return pred_imgs, corrupt_imgs, resid_ints


def get_candid_mask(resid_ints, random_masks, radius=25, n_top=5, res_thr=0.2):
    '''
    determine the masks which potentially inpainted the pathologies
    and find a union of the candidate regions. this union is not necessarly
    covered one single area; in fact it is a combined mask with selected 
    n_top masks and can be distributed to different locations or one single
    area.
    Parameters
    ----------
    resid_ints : list
        intensity differences btw orig and inpainted images.
    random_masks : list
        circular masks for corrupting the image.
    radius : int, optional
        radius of the circular holes. The default is 25.
        used to find out whether the intensity differences with respect to the
        circule area is > or < a certain thr.
    n_top : int, optional
        pick the n_top number of highest intensity differences. The default is 5.
        in case we have several pathologies in one subjects or several circles
        partially inpaint a single pathology.
    res_thr : floating value btw 0 to 1, optional
        The thr mentiond in "radius" arg. The default is 0.2.
        if not satisfied, empty union mask will be returned
        i.e, we either have a union mask or nothing: no pathology detected
        or the model was failed, perhaps due to the very small size of the 
        pathology.
    Returns
    -------
    union_mask : array
        a binary mask representing the final regions to be inpainated.
    '''

    resid_ints = np.array(resid_ints)
    resid_max = np.max(resid_ints)
    circle_area = 3.14 * radius * radius
    # only some slices contains abnormalities
    if (resid_max / circle_area) > res_thr:
        candid_ind = np.argpartition(resid_ints, -n_top)[-n_top:]  # index of top candidates
        random_masks = np.array(random_masks)
        candid_masks = random_masks[candid_ind]
        candid_masks = 1 - candid_masks
        union_mask = np.sum(candid_masks, axis=0).astype('float32')  # indeed it is union mask not intersect
        union_mask[union_mask > 0] = 1
    else:
        union_mask = np.zeros_like(random_masks[0])
    return union_mask


def get_pred_candid(main_img, union_mask, model):
    '''
    Get the candidate mask and ipainted the image
    Parameters
    ----------
    main_img : array
        main image to be inpainted.
    union_mask : array
        the candidate union mask.
    model : TenFlow model
        the loaded model with proper weights.
    Returns
    -------
    corrupt_img : array
        main image corrupted with the union mask.
    my_pred : array
        inpainted image.
    '''

    corrupt_img = deepcopy(main_img)
    mask = (1 - union_mask)
    corrupt_img[mask == 0] = 1
    corrupt_img = np.expand_dims(corrupt_img, axis=0)
    mask = np.expand_dims(mask, axis=0)

    my_pred = model.predict([corrupt_img, mask])
    corrupt_img = np.squeeze(corrupt_img)
    my_pred = np.squeeze(my_pred)

    return corrupt_img, my_pred
