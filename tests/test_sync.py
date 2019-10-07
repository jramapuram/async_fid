import os
import numpy as np

# create some random data to post to both the calls
rv = np.random.rand(10000, 28, 28, 1)

def sync_test(task='mnist'):
    """Creates a sync-fid object and tests random data and the test set.

    :param task: string value for task
    :param data_dir: the directory to store the data
    :returns: nothing
    :rtype: None

    """
    # post two sync-tasks, one with RND data and one with real test data
    f.post(fake_images=rv,
           lbda=lambda s: print("\n[{}]\n\tFID for random data vs. test-set : {}.".format(task, s)),
           dataset_str=task
    )
    f.post(fake_images=f.test_dict[task]/255.,
           lbda=lambda s: print("\tFID for test-set vs. test-set : {}\n".format(s)),
           dataset_str=task
    )

# build the sync-fid object
from fid.fid import SyncFID as FID
f = FID(normalize=True, force_cpu=False)

# Add datasets to the FID object (you probably need a lot of RAM for all of these)
f.add_dataset(dataset_str='mnist', root_folder='./mnist')
# f.add_dataset(dataset_str='permuted', root_folder='./permuted_mnist')
# f.add_dataset(dataset_str='fashion', root_folder='./fashion')
# f.add_dataset(dataset_str='cifar10', root_folder='./cifar10')
# f.add_dataset(dataset_str='cifar100', root_folder='./cifar100')
# f.add_dataset(dataset_str='svhn_full', root_folder='./svhn_full')
# f.add_dataset(dataset_str='svhn_centered', root_folder='./svhn_centered')
# f.add_dataset(dataset_str='binarized_mnist', root_folder='./binarized_mnist')
# f.add_dataset(dataset_str='binarized_omniglot', root_folder='./binarized_omniglot')
# f.add_dataset(dataset_str='binarized_omniglot_burda', root_folder='./binarized_omniglot_burda')

# uncomment to test celeba and imagenet (replacing the directory location)
# f.add_dataset(dataset_str='celeba', root_folder=os.path.join(os.path.expanduser('~'), 'datasets/celeba'))
# f.add_dataset(dataset_str='imagefolder', root_folder=os.path.join(os.path.expanduser('~'),'datasets/imagenet_32x32/ILSVRC/Data/CLS-LOC'))
# f.add_dataset(dataset_str='celeba', root_folder='/datasets/celeba')
# f.add_dataset(dataset_str='image_folder', root_folder='/datasets/imagenet_32x32/ILSVRC/Data/CLS-LOC')

# run the actual tests
sync_test('mnist')
# sync_test('fashion') # run others too!
print("sync tests completed!")
