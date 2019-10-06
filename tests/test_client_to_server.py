import time
import argparse
import rpyc
import numpy as np
from functools import partial

num_successfully_posted = 0
max_tests = 2


def printer(x, test_number):
    """ This is a callback that is asynchronously called by BgServingThread

    :param x: the result from the FID calc
    :param test_number: the test-number, overloaded using functools.partial
    :returns: None
    :rtype: None

    """
    print("FID for test {} is {}".format(test_number, x))
    global num_successfully_posted
    num_successfully_posted += 1 # also increment a glocal variable to this thread!


def run(host, port):
    cfg = {'allow_pickle': True, "sync_request_timeout": 180}
    conn = rpyc.connect(host, port, config=cfg)
    bgsrv = rpyc.BgServingThread(conn)

    # create a FID object for MNIST
    fid = conn.root.FID('mnist', '/datasets/mnist', True, False)

    # post some random data to it which will call-back printer here locally
    for i in range(max_tests):
        print_i = partial(printer, test_number=i)
        fid.post(np.random.rand(10000, 28, 28, 1), print_i)
        print("successfully posted data {}, awaiting response...".format(i))

    # note that we could be continuing execution with our training/test loop here!
    while num_successfully_posted < max_tests:
        time.sleep(1)

    print("test completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID Client Test")
    parser.add_argument("--host", type=str, default="localhost",
        help="Specify the host server endpoint")
    parser.add_argument("-p", "--port", type=int, default=8000,
        help="Specify the port on which the server listens")

    args = parser.parse_args()
    run(host=args.host, port=args.port)
