import os
import time
import numpy as np

# create some random data to post to both the calls
rv = np.random.rand(10000, 28, 28, 1)


def async_test(f, task='mnist'):
    """Creates a sync-fid object and tests random data and the test set.

    :param task: string value for task
    :param data_dir: the directory to store the data
    :returns: nothing
    :rtype: None

    """
    # f.post(fake_images=rv,
    #        lbda=lambda s: print("\n[{}]\tFID for random data vs. test-set : {}.".format(task, s)),
    #        dataset_str=task
    # )
    # print('posted async item!')

    f.post_with_images(fake_images=f.test_dict[task]/255.,
                       real_images=f.test_dict[task],
                       lbda=lambda s: print("\n[{}]\tFID for test-set vs. test-set : {}\n".format(task, s)))
    print('posted async item!')

    f.post_with_images(fake_images=f.test_dict[task],
                       real_images=f.test_dict[task]/255.,
                       lbda=lambda s: print("\n[{}]\tFID for test-set vs. test-set : {}\n".format(task, s)))
    print('posted async item!')

    f.post_with_images(fake_images=f.test_dict[task],
                       real_images=f.test_dict[task],
                       lbda=lambda s: print("\n[{}]\tFID for test-set vs. test-set : {}\n".format(task, s)))
    print('posted async item!')


from fid.fid import AsyncFID as FID
f = FID(normalize=True, force_cpu=False)
f.add_dataset(dataset_str='mnist', root_folder='./mnist')

# Note that the async FID returns here instantly instead of blocking
async_test(f, task='mnist')

# we need to introduce a sleep here to see this message after the garbage mound of TF bla
time.sleep(10)
f.terminate() # kills the inner process **AFTER** finishing the queue of tasks
print("\nasync tests spawned, waiting for spawned process to terminate...")

# join blocks the current thread until f terminates.
f.join()
