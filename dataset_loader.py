import os
import pickle
import numpy as np

from collections import namedtuple
from torchvision import datasets as D
from scipy.misc import imread, imresize, imsave, bytescale


from datasets.loader import get_loader


def load_celeba_data(root_folder, flag='training', side_length=None, num=None):
    """ Helper to load the celeba dataset and return the np array

    :param root_folder: the base path
    :param flag: training or testing?
    :param side_length: wtf is this?
    :param num: upper bound number of images to read
    :returns: dataset in numpy
    :rtype: np.array

    """
    dir_path = os.path.join(root_folder, 'img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break

        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))

    imgs = np.concatenate(imgs, 0)
    return imgs.astype(np.uint8)


def load_celeba140_data(root_folder, flag='training', side_length=None, num=None):
    """ Helper to load the celeba 140x140 center crop dataset and return the np array

    :param root_folder: the base path
    :param flag: training or testing?
    :param side_length: wtf is this?
    :param num: upper bound number of images to read
    :returns: dataset in numpy
    :rtype: np.array

    """
    dir_path = os.path.join(root_folder, 'img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[39:179,19:159]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])

        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break

        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))

    imgs = np.concatenate(imgs, 0)
    return imgs.astype(np.uint8)


# Center crop 140x140 and resize to 64x64
# Consistent with the preporcess in WAE [1] paper
# [1] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf.
# Wasserstein auto-encoders. International Conference on Learning Representations, 2018.
def preprocess_celeba140(root_folder):
    # x_val = load_celeba140_data(root_folder, 'val', 64)
    x_test = load_celeba140_data(root_folder, 'test', 64)
    x_train = load_celeba140_data(root_folder, 'training', 64)
    return x_train, x_test


# Center crop 128x128 and resize to 64x64
def preprocess_celeba(root_folder):
    # x_val = load_celeba_data('val', 64)
    x_test = load_celeba_data(root_folder, 'test', 64)
    x_train = load_celeba_data(root_folder, 'training', 64)
    return x_train, x_test


def data_loader_to_np(data_loader):
    """ Use the data-loader to iterate and return np array.

    :param data_loader: the torch dataloader
    :returns: numpy array of input images
    :rtype: np.array

    """
    images_array = []
    for img, _ in data_loader:
        images_array.append(img)

    images_array = np.transpose(np.vstack(images_array), [0, 2, 3, 1])

    # convert to uint8
    if images_array.max() < 255:
        images_array *= 255

    print("images = ", images_array.shape)
    assert images_array.shape[-1] == 3 or images_array.shape[-1] == 1
    return images_array.astype(np.uint8)


def get_numpy_dataset(task, root_dir):
    """ Builds the loader --> get train and test numpy data and returns.

    :param task: the string task to use
    :param root_dir: the directory to save and load MNIST from
    :returns: two numpy arrays, training and test
    :rtype: (np.array, np.array)

    """
    if task == 'celeba':
        return preprocess_celeba(root_dir)
    elif task == 'celeba140':
        return preprocess_celeba140(root_dir)

    # all other datasets are routed through our standard dataset loader
    Args = namedtuple("Args", "batch_size data_dir task cuda")
    kwargs = {'batch_size': 1, 'data_dir': root_dir, 'task': task, 'cuda': False}
    args = Args(**kwargs)
    loader = get_loader(args, **kwargs)

    # gather the training and test datasets in numpy
    x_train = data_loader_to_np(loader.train_loader)
    x_test = data_loader_to_np(loader.test_loader)

    return x_train, x_test
