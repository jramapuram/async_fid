import os
import time
import numpy as np

# create some random data to post to both the calls
rv = np.random.rand(10000, 28, 28, 1)


def async_test(task='mnist', data_dir='./mnist', force_cpu=False):
    """Creates a sync-fid object and tests random data and the test set.

    :param task: string value for task
    :param data_dir: the directory to store the data
    :returns: nothing
    :rtype: None

    """
    from fid.fid import AsyncFID as FID
    f = FID(task, data_dir, force_cpu=force_cpu)

    f.post(rv, lambda s: print("\n[{}]\tFID for random data vs. test-set : {}.".format(task, s)))
    print('posted async item!')

    f.post(f.test/255., lambda s: print("\n[{}]\tFID for test-set vs. test-set : {}\n".format(task, s)))
    print('posted async item!')

    # return a handle to the FID object to termiante it
    return f


# Note that the async FID returns here instantly instead of blocking
f = async_test(task='mnist', data_dir='./mnist', force_cpu=False)

# we need to introduce a sleep here to see this message after the garbage mound of TF bla
time.sleep(10)
f.terminate() # kills the inner process **AFTER** finishing the queue of tasks
print("\nasync tests spawned, waiting for spawned process to terminate...")

# join blocks the current thread until f terminates.
f.join()
