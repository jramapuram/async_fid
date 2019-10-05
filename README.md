# async_fid

Asynchronous & Synchronous(TF) [FID](https://arxiv.org/abs/1706.08500) calculation helpers.
This repo extends the code that was generously written by [1](https://github.com/daib13/TwoStageVAE/blob/master/fid_score.py) and [2](https://github.com/bioinf-jku/TTUR/blob/master/fid.py) while providing some nice extras & a cleaner interface.


### PyRPC Client-Server Solution

Posting messages to a remote FID-computing server is simple! See `tests/test_client_to_server.py` for an example.

```bash
# replace run_gpu.sh with run_cpu.sh for CPU workloads (not recommended).
./docker/run_gpu.sh "python server.py"                                  # run on the server
./docker/run_gpu.sh "python -m unittest tests/test_client_to_server.py" # run on client
```

### Sync-Tests

To test the sync version run `python -m unittest tests/test_sync.py`, here are the results:

``` bash
[mnist]
        FID for random data vs. test-set : 419.81745204835573.
        FID for test-set vs. test-set : 2.000466499576364e-05
...

[permuted] # actually is doing permuted-mnist
        FID for random data vs. test-set : 336.31173076137065.
        FID for test-set vs. test-set : 1.149141822937758e-05
...

[fashion]
        FID for random data vs. test-set : 402.12138184169527.
        FID for test-set vs. test-set : 7.3852093285004855e-06
...

[cifar10]
        FID for random data vs. test-set : 415.6104467123265.
        FID for test-set vs. test-set : 0.23097138537141707
...

[cifar100]
        FID for random data vs. test-set : 423.52608082034476.
        FID for test-set vs. test-set : 0.27762879099949245
...

[svhn_full]
        FID for random data vs. test-set : 411.5628830393524.
        FID for test-set vs. test-set : 4.3500883367190895
...

[svhn_centered]
        FID for random data vs. test-set : 393.96311903676656.
        FID for test-set vs. test-set : 9.076697602166995
...

[binarized_mnist]
        FID for random data vs. test-set : 446.37204352094204.
        FID for test-set vs. test-set : 3.524291969370097e-12
...

[binarized_omniglot]
        FID for random data vs. test-set : 440.3752313180992.
        FID for test-set vs. test-set : 0.0034667244183284635
...

[binarized_omniglot_burda]
        FID for random data vs. test-set : 400.55568682515616.
        FID for test-set vs. test-set : 0.2629414960831542
...

[celeba]
        FID for random data vs. test-set : 454.7212195100483.
        FID for test-set vs. test-set : 1.2387706087668702
...

[image_folder] # evaluating 32x32 imagenet raw imgs
        FID for random data vs. test-set : 425.07742193926936.
        FID for test-set vs. test-set : 5.2435325640107235
```

### Async-Tests

Similarly run `python -m unittest tests/test_async.py` to test async solution.
