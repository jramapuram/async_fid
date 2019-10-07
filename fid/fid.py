from __future__ import absolute_import, division, print_function

import os
import queue
import dill
import os.path, sys, tarfile
import numpy as np
import tensorflow as tf
import numpy as np
import gzip, pickle
import urllib

from time import sleep
from scipy import linalg
from functools import partial
from six.moves import range, urllib
from multiprocessing import Process, Queue, Event, Manager


from fid.dataset_loader import get_numpy_dataset


def create_inception_graph(path):
    """ Converts serialized graph to tf.Graph

    :param path: file location of serialized graph
    :returns: None, but sets the default tf.Graph to the serialized graph
    :rtype: None

    """
    from tensorflow.python.platform import gfile
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')


def _get_inception_layer(sess):
    """ Prepares inception net for batched usage and returns pool_3 layer.

    :param sess: the tf.Session
    :returns: the pool3 layer
    :rtype: tf.Operation

    """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    return pool3


def get_activations_tf(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)

        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)

    if verbose:
        print(" done")

    return pred_arr


def fid_score(codes_g, codes_r, eps=1e-6):
    """ Provides fid scores give the generated and real pool3 codes.

    :param codes_g: generated pool3 codes
    :param codes_r: reconstructed pool3 codes
    :param eps: tolerance
    :returns: the fid score
    :rtype: float32

    """
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    mn_g = codes_g.mean(axis=0)
    mn_r = codes_r.mean(axis=0)

    cov_g = np.cov(codes_g, rowvar=False)
    cov_r = np.cov(codes_r, rowvar=False)

    covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
    if not np.isfinite(covmean).all():
        cov_g[range(d), range(d)] += eps
        cov_r[range(d), range(d)] += eps
        covmean = linalg.sqrtm(cov_g.dot(cov_r))

    score = np.sum((mn_g - mn_r) ** 2) + (np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
    return score


def preprocess_fake_images(fake_images, norm=False):
    """ 1 Chan --> 3 Chan + normalize (if requested)

    :param fake_images: the numpy array of fake images
    :param norm: bool on whether to normalize and rescale images
    :returns: processed dataset
    :rtype: np.array

    """
    if np.shape(fake_images)[-1] == 1:
        fake_images = np.concatenate([fake_images, fake_images, fake_images], -1)

    if norm:
        for j in range(np.shape(fake_images)[0]):
            fake_images[j] = (fake_images[j] - np.min(fake_images[j])) / (np.max(fake_images[j] - np.min(fake_images[j])))

    fake_images *= 255
    return fake_images[0:10000]


def preprocess_real_images(real_images):
    """ Pre-process real image and return. 1 --> 3 channel

    :param real_images: the numpy array of real images
    :returns: RGB f32 dataset
    :rtype: np.array

    """
    if np.shape(real_images)[-1] == 1:
        real_images = np.concatenate([real_images, real_images, real_images], -1)

    real_images = real_images.astype(np.float32)
    return real_images


def check_or_download_inception():
    """ Checks if inception model is download, otherwise downloads it.

    :returns: path to the file
    :rtype: str

    """
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    model_file = 'classify_image_graph_def.pb'
    if not os.path.exists(model_file):
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)

        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb')

    return str(model_file)


def fid_read_dataset(fake_images, dataset, root_folder, norm=True):
    """ Compare fake_images to ones from test dataset (provided as str).

    :param fake_images: the fake images (np)
    :param dataset: str value for the dataset (autoloaded)
    :param root_folder: the folder for the dataset
    :param norm: whether to normalize or not
    :returns: fid score
    :rtype: float32

    """
    assert isinstance(fake_images, np.ndarray)
    real_images, _ = get_numpy_dataset(dataset, root_folder)
    return fid(fake_images, real_images, root_folder, norm)


def fid(fake_images, real_images, norm=True, sess=None):
    """ Compare fake_images to ones from test dataset.

    :param fake_images: the fake images (numpy)
    :param real_images: the real numpy images
    :param norm: whether to normalize or not
    :param sess: the tensorflow session
    :returns: fid score
    :rtype: float32

    """
    # sanity checks to not compute fid on erroneous data
    assert isinstance(fake_images, np.ndarray) and isinstance(real_images, np.ndarray), \
        print('need np arrays, got fake = ', type(fake_images), " | real = ", type(real_images))
    assert fake_images.min() >= 0 and fake_images.max() <= 1.0, \
        "fake images need to be in [0, 1] range."
    assert real_images.min() >= 0 and real_images.max() <= 255.0 and np.sum(real_images > 1) >= 1, \
        "real images need to be in [0, 255] range."

    np.random.shuffle(real_images)
    real_images = real_images[0:10000]
    real_images = preprocess_real_images(real_images)
    fake_images = preprocess_fake_images(fake_images, norm)

    if sess is None: # download inception if it doesnt exist
        inception_path = check_or_download_inception()
        create_inception_graph(inception_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # calculate the FID features
            real_out = get_activations_tf(real_images, sess)
            fake_out = get_activations_tf(fake_images, sess)
    else:
        # calculate the FID features
        real_out = get_activations_tf(real_images, sess)
        fake_out = get_activations_tf(fake_images, sess)

    # return the FID score
    return fid_score(real_out, fake_out)


class SyncFID(object):
    def __init__(self, normalize=True, force_cpu=False):
        """ Helper object that sync posts to a queue to compute the FID.

        :param normalize: normalize images or not?
        :param force_cpu: force the session to be on the CPU
        :returns: FID object
        :rtype: object

        """
        self.force_cpu = force_cpu
        self.normalize = normalize
        self.test_dict = {}

        # setup TF and session
        tf.reset_default_graph()
        inception_path = check_or_download_inception()
        create_inception_graph(inception_path)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}) if force_cpu else tf.ConfigProto())
        self.sess.run(tf.global_variables_initializer())

    def _process(self, fake_images, lbda, dataset_str):
        """ Internal member to compute FID. Adds a dataset if needed.

        :param fake_images: the numpy fake images
        :param lbda: the lambda function
        :param dataset_str: the name of the dataset
        :returns: None, calls the internal lambda
        :rtype: None

        """
        if dataset_str not in self.test_dict:
            print("Error! {} not in test dictionary, add via add_dataset call.")
            return

        # pick the right dataset
        test_set = self.test_dict[dataset_str]

        # compute the FID
        fid_score = fid(fake_images, test_set,
                        norm=self.normalize,
                        sess=self.sess)

        # call the lambda
        lbda(fid_score)

    def add_dataset(self, dataset_str, root_folder):
        """ Adds the dataset to the test container

        :param dataset_str: the name of the dataset
        :param root_folder: where the dataset is stored or should be downloaded
        :returns: nothing, but adds to member dict
        :rtype: None

        """
        if dataset_str is not None and dataset_str not in self.test_dict:
            _, test = get_numpy_dataset(dataset_str, root_folder)
            self.test_dict[dataset_str] = test

        return self.test_dict[dataset_str]

    def post(self, fake_images, lbda, dataset_str):
        """ given a set of fake images and a lambda or fn that accepts the score compute FID

        :param fake_images: fake images, TODO: this should be 10k I believe
        :param lbda: any lambda or fn that accepts the fid score as ONLY param
        :param dataset_str: specify which dataset's test set to use
        :returns: nothing, just calls the lambda on the fid score
        :rtype: None

        """
        return self._process(fake_images, lbda, dataset_str=dataset_str)


class AsyncFID(object):
    def __init__(self, normalize=True, force_cpu=False):
        """ Helper object that async posts to a queue to compute the FID.

        :param normalize: normalize images or not?
        :param force_cpu: force the session to be on the CPU
        :returns: FID object
        :rtype: object

        """
        self.normalize = normalize
        self.force_cpu = force_cpu
        self.manager = Manager()
        self.test_dict = self.manager.dict() # house all the test datasets

        # force the session to be on the CPU
        self.session_conf = tf.ConfigProto(device_count={'GPU':0}) if force_cpu else tf.ConfigProto()

        # build a thread-safe queue and start an event loop
        self.is_running = Event()
        self.q = Queue()
        self.event_loop = Process(target=self._loop, args=(self.q,))
        self.event_loop.start()

    def join(self):
        """ Blocks the calling process.

        :returns: None
        :rtype: None

        """
        self.event_loop.join()

    def terminate(self):
        """ Terminates the event loop.

        :returns: None
        :rtype: None

        """
        self.is_running.set()

    def add_dataset(self, dataset_str, root_folder):
        """ Adds the dataset to the test container

        :param dataset_str: the name of the dataset
        :param root_folder: where the dataset is stored or should be downloaded
        :returns: nothing, but adds to member dict
        :rtype: None

        """
        if dataset_str is not None and dataset_str not in self.test_dict:
            _, test = get_numpy_dataset(dataset_str, root_folder)
            self.test_dict[dataset_str] = test

        return self.test_dict[dataset_str]

    def _loop(self, q):
        """ Internal member to pop queue and compute FID

        :param q: the shared thread safe queue
        :returns: None, calls the internal lambda
        :rtype: None

        """
        # build the graph for inceptionv3
        tf.reset_default_graph()
        inception_path = check_or_download_inception()
        create_inception_graph(inception_path)

        with tf.Session(config=self.session_conf) as sess:
            sess.run(tf.global_variables_initializer())

            while not self.is_running.is_set():
                fake_images, pkld, dataset_str = None, None, None

                while not q.empty(): # process entire queue before exiting
                    try: # block for 1 sec and catch the empty exception
                        fake_images, pkld, dataset_str = q.get(block=True, timeout=1)
                    except queue.Empty:
                        continue

                    # process the item popped (only if there was one popped)
                    if fake_images is not None and pkld is not None and dataset_str is not None:
                        if dataset_str not in self.test_dict:
                            print("Error! {} not in test dictionary, add via add_dataset call.".format(dataset_str))
                        else:
                            # pick the right dataset and compute the FID
                            test_set = self.test_dict[dataset_str]
                            fid_score = fid(fake_images, test_set,
                                            norm=self.normalize,
                                            sess=sess)

                            # load the pickled lambda function and execute
                            lbda = dill.loads(pkld)
                            lbda(fid_score)

                sleep(1)

    def post(self, fake_images, lbda, dataset_str):
        """ given a set of fake images and a lambda or fn that accepts the score compute FID

        :param fake_images: fake images, TODO: this should be 10k I believe
        :param lbda: any lambda or fn that accepts the fid score as ONLY param
        :param dataset_str: the str name of the dataset
        :returns: nothing, just calls the lambda on the fid score
        :rtype: None

        """
        self.q.put((fake_images,
                    dill.dumps(lbda),
                    dataset_str))
