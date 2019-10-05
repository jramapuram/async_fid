import os
import numpy as np

# create some random data to post to both the calls
rv = np.random.rand(10000, 28, 28, 1)

def sync_test(task='mnist', data_dir='./mnist'):
    """Creates a sync-fid object and tests random data and the test set.

    :param task: string value for task
    :param data_dir: the directory to store the data
    :returns: nothing
    :rtype: None

    """
    from fid.fid import SyncFID as FID
    f = FID(task, data_dir)
    f.post(rv, lambda s: print("\n[{}]\n\tFID for random data vs. test-set : {}.".format(task, s)))
    f.post(f.test/255., lambda s: print("\tFID for test-set vs. test-set : {}\n".format(s)))

# Run all the things
sync_test(task='mnist', data_dir='./mnist')
sync_test(task='permuted', data_dir='./permuted_mnist')
sync_test(task='fashion', data_dir='./fashion')
sync_test(task='cifar10', data_dir='./cifar10')
sync_test(task='cifar100', data_dir='./cifar100')
sync_test(task='svhn_full', data_dir='./svhn_full')
sync_test(task='svhn_centered', data_dir='./svhn_centered')
sync_test(task='binarized_mnist', data_dir='./binarized_mnist')
sync_test(task='binarized_omniglot', data_dir='./binarized_omniglot')
sync_test(task='binarized_omniglot_burda', data_dir='./binarized_omniglot_burda')

# uncomment to test celeba and imagenet (replacing the directory location)
# sync_test(task='celeba', data_dir=os.path.join(os.path.expanduser('~'), 'datasets/celeba'))
# sync_test(task='imagefolder', data_dir=os.path.join(os.path.expanduser('~'),'datasets/imagenet_32x32/ILSVRC/Data/CLS-LOC'))
# sync_test(task='celeba', data_dir='/datasets/celeba')
# sync_test(task='image_folder', data_dir='/datasets/imagenet_32x32/ILSVRC/Data/CLS-LOC')
print("sync tests completed!")
