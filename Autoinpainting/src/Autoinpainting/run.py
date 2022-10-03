import argparse
import os
import tensorflow as tf

from WriteBackNifti import image_to_nifti
from autoinpaint import self_inpaint
from data_prepare import prepare_2d
from libs.paths_dirs_stuff import get_sub_dirs, creat_dir
from libs.unet_model import InpaintingUnet

parser = argparse.ArgumentParser(description='Anomaly detection with Autoinpainting')
parser.add_argument('--checkpoint_dir', type=str, help='Dir to model weights', required=True)
parser.add_argument('--image_size', type=int, default=256, help='2D img size', required=False)
parser.add_argument('--upper_cut', type=int, default=1, help='skip upper slices', required=True)
parser.add_argument('--lower_cut', type=int, default=1, help='skip buttom slices', required=True)
parser.add_argument('--input_dir', type=str, help='Input directory to the nifti volumes', required=True)
parser.add_argument('--output_dir', type=str, help='Output directory to save results and logs', required=True)
parser.add_argument('--n_candid', type=int, default=5, help='number of candid circles', required=False)
parser.add_argument('--circle_rad', type=int, default=19, help='radius of moving circles', required=False)
parser.add_argument('--thr', type=float, default=0.06, help='normalized thr by circle size', required=False)
parser.add_argument('--max_bound_map', type=int, default=9, help='Binary map upper bound', required=False)
parser.add_argument('--min_bound_map', type=int, default=4, help=' Binary map lower bound', required=False)
parser.add_argument('--interval_idx', type=int, default=8, help='distance btw moving circles', required=False)
parser.add_argument('--exp_name', type=str, default='AutoPet', help='set a name for experiment', required=False)
parser.add_argument('--n_top_paint', type=int, default=3, help='number of top inpainting circles', required=False)
parser.add_argument('--max_bound_ct', type=int, default=800, help='CT histogram windowing upper bound', required=False)
parser.add_argument('--min_bound_pet', type=int, default=0, help='PET histogram windowing lower bound', required=False)
parser.add_argument('--max_bound_pet', type=int, default=12, help='PET histogram windowing upper bound', required=False)
parser.add_argument('--min_bound_ct', type=int, default=-1000, help='CT histogram windowing lower bound',
                    required=False)
parser.add_argument('--pet_pattern', type=str, default="_PET.nii.gz", help='Filename patterns of PET nifti volumes',
                    required=False)
parser.add_argument('--ct_pattern', type=str, default="_CT_downsampled.nii.gz",
                    help='Filename patterns of CT nifti volumes', required=False)

args = parser.parse_args()
res_thr = args.thr
rad = args.circle_rad
exp_name = args.exp_name
nifti_root = args.input_dir
write_dir = args.output_dir
ct_pattern = args.ct_pattern
image_size = args.image_size
pet_pattern = args.pet_pattern
n_top_inpaint = args.n_top_paint
interval_idx = args.interval_idx
min_bound_ct = args.min_bound_ct
max_bound_ct = args.max_bound_ct
min_bound_pet = args.min_bound_pet
max_bound_pet = args.max_bound_pet
min_bound_map = args.min_bound_map
max_bound_map = args.max_bound_map
n_top_circle_select = args.n_candid
crop_vol_ind_buttom = args.lower_cut
crop_vol_ind_top = args.upper_cut
crop_vol_ind_top = (-1) * crop_vol_ind_top

# step1: prepare 2D images
write_dir = os.path.join(write_dir, exp_name)
img_dir = os.path.join(write_dir, '2D/input/imgs')
map_dir = os.path.join(write_dir, '2D/input/maps')
nifti_subfolders = get_sub_dirs(nifti_root)[1:]
n_subject = len(nifti_subfolders)

prepare_2d(nifti_subfolders, img_dir, map_dir, ct_pattern, pet_pattern,
           min_bound_ct, max_bound_ct, min_bound_pet, max_bound_pet,
           min_bound_map, max_bound_map, image_size, write_dir, nifti_root)

# step2: autoinpainting the 2D images

checkpoint_dir = args.checkpoint_dir

output_dir = os.path.join(write_dir, '2D/output/results')
creat_dir(output_dir)
imgs_dir = os.path.join(write_dir, '2D/input/imgs')
maps_dir = os.path.join(write_dir, '2D/input/maps')
checkpoint_multi = tf.train.latest_checkpoint(checkpoint_dir)
img_subjects = get_sub_dirs(imgs_dir)[1:]
map_subjects = get_sub_dirs(maps_dir)[1:]
model_multi = InpaintingUnet(conv_layer='gconv', load_weights=checkpoint_multi, train_bn=True)

n_subjects = len(img_subjects)

self_inpaint(img_subjects, map_subjects, n_subjects, output_dir,
             image_size, crop_vol_ind_buttom, crop_vol_ind_top,
             n_top_inpaint, rad, interval_idx, n_top_circle_select,
             model_multi, res_thr, checkpoint_dir)

# step3: calculating the residuals and writing them as .nii files

nifti_subfolders = get_sub_dirs(nifti_root)[1:]

output_img_subfolder = get_sub_dirs(output_dir)[1:]
n_subject_nifti = len(nifti_subfolders)
n_subject_input = len(img_subjects)
n_subject_output = len(output_img_subfolder)

write_nifti_root = os.path.join(write_dir, '3d_nifti')
write_path_ct_res = os.path.join(write_nifti_root, 'ct_residual')
write_path_pet_res = os.path.join(write_nifti_root, 'pet_residual')
creat_dir(write_path_ct_res)
creat_dir(write_path_pet_res)

image_to_nifti(n_subject_nifti, n_subject_input, n_subject_output,
               nifti_subfolders, img_subjects, output_img_subfolder,
               ct_pattern, write_path_ct_res, write_path_pet_res)
