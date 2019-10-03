import os
import numpy as np

# create some random data to post to both the calls
rv = np.random.rand(10000, 28, 28, 1)

def async_test(task='mnist', data_dir='./mnist'):
    """Creates a sync-fid object and tests random data and the test set.

    :param task: string value for task
    :param data_dir: the directory to store the data
    :returns: nothing
    :rtype: None

    """
    from fid import AsyncFID as FID
    f = FID(task, data_dir)
    f.post(rv, lambda s: print("\n[{}]\tFID for random data vs. test-set : {}.".format(task, s)))
    print('posted async item!')
    f.post(f.test/255., lambda s: print("\n[{}]\tFID for test-set vs. test-set : {}\n".format(task, s)))
    print('posted async item!')


# Run all the things
async_test(task='mnist', data_dir='./mnist')
async_test(task='fashion', data_dir='./fashion')
print("async tests completed, waiting for spawned process to terminate...", sep="")
for i in range(200):
    print(".", sep="")
