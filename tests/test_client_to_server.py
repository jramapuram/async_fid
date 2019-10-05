import time
import rpyc
import numpy as np
from functools import partial

num_successfully_posted = 0
max_tests = 20

def printer(x, test_number):
    print("FID for test {} is {}".format(test_number, x))
    global num_successfully_posted
    num_successfully_posted += 1

cfg = {'allow_pickle': True, "sync_request_timeout": 180}
conn = rpyc.connect("localhost", 8000, config=cfg)
bgsrv = rpyc.BgServingThread(conn)

# create a FID object for MNIST
fid = conn.root.FID('mnist', '/datasets/mnist', True, False)

# post some random data to it which will call-back printer here locally
for i in range(max_tests):
    print_i = partial(printer, test_number=i)
    fid.post(np.random.rand(10000, 28, 28, 1), print_i)
    print("successfully posted data {}, awaiting response...".format(i))

while num_successfully_posted < max_tests:
    time.sleep(1)

print("test completed successfully!")
