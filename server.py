#!/usr/bin/env python3

import time
import queue
import rpyc
rpyc.core.channel.Channel.COMPRESSION_LEVEL = 0
rpyc.core.stream.SocketStream.MAX_IO_CHUNK = 65355*10

import argparse
from threading import Thread
from multiprocessing import Event


from fid.fid import SyncFID


class FIDService(rpyc.Service):
    class FID(object):

        class __FID(object):
            def __init__(self, dataset_str, root_folder, normalize=True, force_cpu=False):
                self.dataset_str = dataset_str
                self.root_folder = root_folder
                self.normalize = normalize
                self.force_cpu = force_cpu

                # build the central FID object
                self.fid = SyncFID(dataset_str=self.dataset_str, root_folder=self.root_folder,
                                       normalize=self.normalize, force_cpu=self.force_cpu)

                # hold the fake images / lambda
                self.q = queue.Queue()
                self.is_running = Event()
                self.worker = Thread(target=self.work)
                self.worker.start()

            def terminate(self):
                """ Terminates the event loop.

                :returns: None
                :rtype: None

                """
                self.is_running.set()

            def _check_for_new_dataset(self, **kwargs):
                """ Helper to add a test dataset to the FID object

                :returns: Nothing, but adds the dataset
                :rtype: None

                """
                dataset_str = kwargs.get('dataset_str', None)
                root_folder = kwargs.get('root_folder', self.dataset_str)
                normalize = kwargs.get('normalize', self.normalize)
                force_cpu = kwargs.get('force_cpu', self.force_cpu)

                # if we detect a change add that dataset to the FID object
                if dataset_str is not None and dataset_str != self.dataset_str:
                    self.fid.add_dataset(dataset_str=dataset_str, root_folder=root_folder,
                                         normalize=normalize, force_cpu=force_cpu)

            def work(self):
                """ Internal worker.

                :returns: None
                :rtype: None

                """
                while not self.is_running.is_set(): # flag to end worker
                    while not self.q.empty():       # process entire queue before exiting
                        fake_images, lbda = None, None
                        try: # block for 1 sec and catch the empty exception
                            fake_images, lbda = self.q.get(block=True, timeout=1)
                        except queue.Empty:
                            continue

                        if fake_images is not None and lbda is not None:
                            async_lbda = rpyc.async_(lbda)
                            self.fid.post(fake_images=rpyc.classic.obtain(fake_images),
                                      lbda=async_lbda)

                    time.sleep(1)

            def post(self, fake_images, lbda, **kwargs):
                """ Posts a set of fake images with a lambda function to operate over FID score

                :param fake_images: the set of numpy fake images
                :param lbda: a lambda function taking 1 param as input
                :returns: None
                :rtype: None

                """
                # TODO: add a dataset if it isn't being monitored
                # self._check_for_new_dataset(**kwargs)
                self.q.put((fake_images, lbda))

        # Singleton to return only one instance
        instance = None

        def __init__(self, dataset_str, root_folder, normalize=True, force_cpu=False):
            if not FIDService.FID.instance:
                FIDService.FID.instance = FIDService.FID.__FID(
                    dataset_str, root_folder, normalize=True, force_cpu=False
                )
            # else:
            #     FID.instance._check_for_new_dataset(dataset_str=dataset_str,
            #                                         root_folder=root_folder,
            #                                         normalize=True,
            #                                         force_cpu=False)

        def __getattr__(self, name):
            return getattr(self.instance, name)


def run(port):
    print("starting server on port {}...".format(port))
    from rpyc.utils.server import ThreadedServer
    cfg = {'allow_all_attrs': True, 'allow_pickle': True}
    ThreadedServer(FIDService, port=port, protocol_config=cfg).start()
    print("shutting down server on port {}...".format(port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FID HTTP server")
    parser.add_argument("-p", "--port", type=int, default=8000,
        help="Specify the port on which the server listens")
    args = parser.parse_args()
    run(port=args.port)
