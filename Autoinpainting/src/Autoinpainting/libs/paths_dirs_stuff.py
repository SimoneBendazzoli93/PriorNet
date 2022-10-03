import os
import re

_nsre = re.compile('([0-9]+)')


def natural_sort_key(s):
    '''
    sorting the strings bases on 0:9 numbers and alphabets
    '''
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def creat_dir(my_path):
    '''
    Parameters
    ----------
    my_path : str
        a directory like c:/mydata/.

    Returns
    -------
    create a directory if does not exist.

    '''
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    else:
        pass


def get_data_list(data_path, pattern='.png'):
    '''
    Parameters
    ----------
    weight_path : str
        directory where the data are stored.
    pattern : str, optional
        the extensionfile pattern that a file can be recognized with.
        The default is '.h5' for model weights and .png for images.

    Returns
    -------
    file_list : sorted list
        a sorted list of data. In case of model weights the last item is
        the last weights.

    '''
    file_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    file_list.sort(key=natural_sort_key)
    return file_list


def get_sub_dirs(data_path):
    '''
    take the main dir and return the sub dirs
    ----------
    data_path : str
        path to main dir.

    Returns
    -------
    sub_dirs : list
        full path to sub dirs.

    '''
    sub_dirs = []
    for roots, dirs, files in os.walk(data_path):
        sub_dirs.append(roots)
    sub_dirs.sort(key=natural_sort_key)
    return sub_dirs


def get_n_data(data_path):
    '''
    number of images inside the train/test dirs
    '''
    filepath = get_sub_dirs(data_path)[1]
    filenames = [f for f in os.listdir(filepath)]

    return len(filenames)
