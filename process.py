import SimpleITK
import os
import json
from pathlib import Path
import pickle
import numpy as np
import scipy.special
from export import save_segmentation_nifti_from_softmax
import onnxruntime
from numba import cuda
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import tensorflow as tf
from batchgenerators.augmentations.utils import pad_nd_image

from preprocessing import GenericPreprocessor

isfile = os.path.isfile
from copy import deepcopy
import shutil

join = os.path.join


class Priornet():
    def __init__(self):
        self.input_path = '/input'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'
        self.output_nifti_path = '/output_nifti'
        self.nii_path = "/input_nifti/imagesTs"
        self.exp_name = "AutoPET"
        self.PATA_output_dir = '/input_nifti/'
        self.ct_pattern = "_CT_downsampled.nii.gz"
        self.pet_pattern = "_PET.nii.gz"
        self.image_size = 256

        self.min_bound_ct = -1000
        self.max_bound_ct = 800
        self.min_bound_pet = 0
        self.max_bound_pet = 12
        self.min_bound_map = 4
        self.max_bound_map = 9
        self.PATA_trained_model = "/opt/trained_models/PATA/autopet_weights/weight"

        self.crop_vol_ind_buttom = 20
        self.crop_vol_ind_top = 20
        self.crop_vol_ind_top = (-1) * self.crop_vol_ind_top
        self.n_top_inpaint = 3
        self.rad = 19
        self.interval_idx = 8
        self.n_top_circle_select = 5
        self.res_thr = 0.06

        self.nnunet_trained_model = "/opt/trained_models/nnUNet/AutoPET_onnx_models"
        self.run_tta = False
        self.nnunet_input_folder = "/input_nifti/nnunet/imagesTs"

    def internal_maybe_mirror_and_pred_3D(self, net, x, n_classes, mirror_axes: tuple,
                                          do_mirroring: bool = True,
                                          mult: np.ndarray = None) -> np.ndarray:

        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        result_torch = np.zeros([1, n_classes] + list(x.shape[2:]),
                                dtype=np.float32)

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = net.run([], {'input': x})[0]
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (4,))})[0]
                result_torch += 1 / num_results * np.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (3,))})[0]
                result_torch += 1 / num_results * np.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (4, 3))})[0]
                result_torch += 1 / num_results * np.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (2,))})[0]
                result_torch += 1 / num_results * np.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (4, 2))})[0]
                result_torch += 1 / num_results * np.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (3, 2))})[0]
                result_torch += 1 / num_results * np.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = net.run([], {'input': np.flip(x, (4, 3, 2))})[0]
                result_torch += 1 / num_results * np.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def get_gaussian(self, patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    def compute_steps_for_sliding_window(self, patch_size, image_size, step_size: float):
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                     zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):

            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def predict(self, net, x, plans, run_tta=True, task=0):
        num_classes = plans['num_classes'] + 1
        stage_plans = plans['plans_per_stage'][1]

        tta = run_tta

        step_size = 0.5
        do_mirroring = tta
        mirror_axes = (0, 1, 2)

        pad_border_mode = 'constant'
        pad_kwargs = {'constant_values': 0}
        regions_class_order = None

        patch_size = np.array(stage_plans['patch_size']).astype(int)

        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape

        steps = self.compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        gaussian_importance_map = self.get_gaussian(patch_size, sigma_scale=1. / 8)

        add_for_nb_of_preds = gaussian_importance_map

        aggregated_results = np.zeros([num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([num_classes] + list(data.shape[1:]), dtype=np.float32)

        with tqdm(total=num_tiles) as pbar:
            for x in steps[0]:
                lb_x = x
                ub_x = x + patch_size[0]
                for y in steps[1]:
                    lb_y = y
                    ub_y = y + patch_size[1]
                    for z in steps[2]:
                        lb_z = z
                        ub_z = z + patch_size[2]

                        predicted_patch = self.internal_maybe_mirror_and_pred_3D(
                            net, data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], num_classes, mirror_axes, do_mirroring,
                            gaussian_importance_map)[0]

                        aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                        aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
                        pbar.update(1)

        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        class_probabilities = aggregated_results / aggregated_nb_of_predictions
        regression = False

        if regions_class_order is None and not regression:
            predicted_segmentation = class_probabilities.argmax(0)
        elif regression:
            predicted_segmentation = class_probabilities
        else:
            class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        return predicted_segmentation, class_probabilities

    def subdirs(self, folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
        if join:
            l = os.path.join
        else:
            l = lambda x, y: y
        res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
               and (prefix is None or i.startswith(prefix))
               and (suffix is None or i.endswith(suffix))]
        if sort:
            res.sort()
        return res

    def subfiles(self, folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
        if join:
            l = os.path.join
        else:
            l = lambda x, y: y
        res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
               and (prefix is None or i.startswith(prefix))
               and (suffix is None or i.endswith(suffix))]
        if sort:
            res.sort()
        return res

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):

        img = SimpleITK.ReadImage(str(mha_input_path))

        SimpleITK.WriteImage(img, str(nii_out_path), True)

    def check_input_folder_and_return_caseIDs(self, input_folder, expected_num_modalities):
        print("This model expects %d input modalities for each image" % expected_num_modalities)
        files = self.subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

        maybe_case_ids = np.unique([i[:-12] for i in files])

        remaining = deepcopy(files)
        missing = []

        assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

        for c in maybe_case_ids:
            for n in range(expected_num_modalities):
                expected_output_file = c + "_%04.0d.nii.gz" % n
                if not isfile(join(input_folder, expected_output_file)):
                    missing.append(expected_output_file)
                else:
                    remaining.remove(expected_output_file)

        print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
              np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
        print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

        if len(remaining) > 0:
            print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
                  np.random.choice(remaining, min(len(remaining), 10)))

        if len(missing) > 0:
            print("Some files are missing:")
            print(missing)
            raise RuntimeError("missing files in input_folder")

        return maybe_case_ids

    def load_inputs(self):

        ct_mha_list = os.listdir(os.path.join(self.input_path, 'images/ct/'))
        pet_mha_list = os.listdir(os.path.join(self.input_path, 'images/pet/'))

        uuid_ct_list = [os.path.splitext(ct_mha)[0] for ct_mha in ct_mha_list]
        uuid_pet_list = [os.path.splitext(pet_mha)[0] for pet_mha in pet_mha_list]
        for uuid_ct, uuid_pet in zip(uuid_ct_list, uuid_pet_list):
            Path(self.nii_path).joinpath(uuid_ct).mkdir(parents=True, exist_ok=True)
            self.convert_mha_to_nii(Path(self.input_path).joinpath("images", "pet", "{}.mha".format(uuid_pet)),
                                    Path(self.nii_path).joinpath(uuid_ct, "{}_PET.nii.gz".format(uuid_ct)))
            self.convert_mha_to_nii(Path(self.input_path).joinpath("images", "ct", "{}.mha".format(uuid_ct)),
                                    Path(self.nii_path).joinpath(uuid_ct, "{}_CT_downsampled.nii.gz".format(uuid_ct)))

        return uuid_ct_list, uuid_pet_list

    def prepare_patient_folder_for_inference(self, patient_folder, config_dict, output_folder):
        for idx, modality in enumerate(config_dict["Modalities"]):
            input_file = Path(patient_folder).joinpath(Path(patient_folder).name + modality)
            output_file = Path(output_folder).joinpath(
                Path(patient_folder).name + "_{0:04d}".format(idx) + "{}".format(config_dict["FileExtension"]))
            shutil.copy(input_file, output_file)

    def preprocess(self, plans, input_files):
        stage_plans = plans['plans_per_stage'][1]
        normalization_schemes = plans['normalization_schemes']
        use_mask_for_norm = plans['use_mask_for_norm']
        transpose_forward = plans['transpose_forward']
        intensity_properties = plans['dataset_properties']['intensityproperties']

        preprocessor = GenericPreprocessor(normalization_schemes, use_mask_for_norm,
                                           transpose_forward, intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             stage_plans[
                                                                 'current_spacing'])

        return d, s, properties

    def run_PATA_prediction(self):

        write_dir = os.path.join(self.PATA_output_dir, self.exp_name)
        img_dir = os.path.join(write_dir, '2D/input/imgs')
        map_dir = os.path.join(write_dir, '2D/input/maps')

        nifti_subfolders = [os.path.join(self.nii_path, i.name) for i in Path(self.nii_path).iterdir() if i.is_dir()]

        print("Start preprocessing for PATA...")
        prepare_2d(nifti_subfolders, img_dir, map_dir, self.ct_pattern, self.pet_pattern,
                   self.min_bound_ct, self.max_bound_ct, self.min_bound_pet, self.max_bound_pet,
                   self.min_bound_map, self.max_bound_map, self.image_size, write_dir, self.nii_path)
        print('Start PATA prediction')
        checkpoint_dir = self.PATA_trained_model
        output_dir = os.path.join(write_dir, '2D/output/results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_multi = tf.train.latest_checkpoint(checkpoint_dir)
        img_subjects = get_sub_dirs(img_dir)[1:]
        map_subjects = get_sub_dirs(map_dir)[1:]
        model_multi = PATAUnet(conv_layer='gconv', load_weights=checkpoint_multi, train_bn=True)

        n_subjects = len(img_subjects)

        run_PATA(img_subjects, map_subjects, n_subjects, output_dir,
                 self.image_size, self.crop_vol_ind_buttom, self.crop_vol_ind_top,
                 self.n_top_inpaint, self.rad, self.interval_idx, self.n_top_circle_select,
                 model_multi, self.res_thr, checkpoint_dir)

        nifti_subfolders = get_sub_dirs(self.nii_path)[1:]

        output_img_subfolder = get_sub_dirs(output_dir)[1:]
        n_subject_nifti = len(nifti_subfolders)
        n_subject_input = len(img_subjects)
        n_subject_output = len(output_img_subfolder)

        write_nifti_root = os.path.join(self.nii_path)
        write_path_ct_res = write_nifti_root
        write_path_pet_res = write_nifti_root

        image_to_nifti(n_subject_nifti, n_subject_input, n_subject_output,
                       nifti_subfolders, img_subjects, output_img_subfolder,
                       self.ct_pattern, write_path_ct_res, write_path_pet_res)

        device = cuda.get_current_device()
        device.reset()

    def run_nnUNet_prediction(self):
        folds = ["0", "1", "2", "3", "4"]

        config_file = [os.path.join(str(Path(self.nnunet_trained_model)), i.name) for i in
                       Path(str(Path(self.nnunet_trained_model))).iterdir() if i.is_file() and (
                           i.name.endswith(".json"))][0]

        with open(config_file) as json_file:
            config_dict = json.load(json_file)

        with open(Path(self.nnunet_trained_model).joinpath("plans.pkl"), 'rb') as file:
            plans = pickle.load(file)

        input_folder = self.nii_path

        patient_dirs = [os.path.join(input_folder, i.name) for i in Path(input_folder).iterdir() if i.is_dir()]

        Path(self.nnunet_input_folder).mkdir(parents=True, exist_ok=True)

        for patient_dir in patient_dirs:
            self.prepare_patient_folder_for_inference(patient_dir, config_dict, self.nnunet_input_folder)

        ids = self.check_input_folder_and_return_caseIDs(self.nnunet_input_folder, len(config_dict["Modalities"]))

        output_files = []

        for patient_id in ids:
            output_filename = Path(self.output_nifti_path).joinpath(patient_id + ".nii.gz")
            if Path(output_filename).is_file():
                continue
            print("Running case {}".format(patient_id))

            output_softmax_filename = None
            output_files.append(output_filename)

            input_files = self.subfiles(self.nnunet_input_folder, prefix=patient_id, sort=True)
            x, _, prop = self.preprocess(plans, input_files)

            model_files = [str(Path(self.nnunet_trained_model).joinpath("fold_{}.onnx".format(fold))) for fold in
                           folds]

            for idx, model_file in enumerate(model_files):
                ort_session = onnxruntime.InferenceSession(model_file,
                                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                pred, softmax_out = self.predict(ort_session, x, plans, run_tta=self.run_tta)

                if idx == 0:
                    softmax = softmax_out
                else:
                    softmax += softmax_out

            softmax /= len(model_files)

            softmax = scipy.special.softmax(softmax, axis=0)
            transpose_forward = plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = plans.get('transpose_backward')
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            if 'segmentation_export_params' in plans.keys():
                force_separate_z = plans['segmentation_export_params']['force_separate_z']
                interpolation_order = plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0

            region_class_order = None

            npz_file = None

            softmax_output = save_segmentation_nifti_from_softmax(softmax, str(output_filename), prop,
                                                                  interpolation_order,
                                                                  region_class_order,
                                                                  None, None,
                                                                  npz_file, None, force_separate_z,
                                                                  interpolation_order_z,
                                                                  out_softmax_fname=output_softmax_filename)

            img = SimpleITK.ReadImage(str(output_filename))
            SimpleITK.WriteImage(img, str(Path(self.output_path).joinpath("{}.mha".format(patient_id))))

    def process(self):

        print('Start processing')
        Path(self.output_nifti_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        _, _ = self.load_inputs()

        self.run_PATA_prediction()
        self.run_nnUNet_prediction()


if __name__ == "__main__":
    Priornet().process()
