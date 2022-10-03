import json
import os


def config_summary(TRAIN_DIR, VAL_DIR, csv_name, BATCH_SIZE, learn_rate,
                   random_seed, conv_operator, batch_norm_flag, val_monitor,
                   n_epochs, exp_name, img_size):
    configs = {}
    configs['TRAIN_DIR'] = TRAIN_DIR
    configs['VAL_DIR'] = VAL_DIR
    configs['csv_name'] = csv_name
    configs['BATCH_SIZE'] = BATCH_SIZE
    configs['learn_rate'] = learn_rate
    configs['random_seed'] = random_seed
    configs['conv_operator'] = conv_operator
    configs['batch_norm_flag'] = batch_norm_flag
    configs['val_monitor_metric'] = val_monitor
    configs['training_epochs'] = n_epochs
    configs['experiment_name'] = exp_name
    configs['image_size'] = img_size

    return configs


def save_json(write_dir, filename, dict_summary):
    json_dir = os.path.join(write_dir, filename)
    with open(json_dir, 'w') as fp:
        json.dump(dict_summary, fp, indent=4)


def load_json(json_path):
    with open(json_path, 'r') as json_file:
        configs = json.load(json_file)
    return configs
