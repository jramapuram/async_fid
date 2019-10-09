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
            def __init__(self, normalize=True, force_cpu=False):
                self.normalize = normalize
                self.force_cpu = force_cpu

                # build the central FID object
                self.fid = SyncFID(normalize=self.normalize,
                                   force_cpu=self.force_cpu)

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

            def add_dataset(self, dataset_str, root_folder):
                """ Helper to add a test dataset to the FID object.

                :param dataset_str: name (str) of the dataset
                :param root_folder: where it is stored
                :returns: nothing, but adds the dataset
                :rtype: None

                """
                self.fid.add_dataset(dataset_str, root_folder)

            def work(self):
                """ Internal worker.

                :returns: None
                :rtype: None

                """
                while not self.is_running.is_set(): # flag to end worker
                    while not self.q.empty():       # process entire queue before exiting
                        fake_images, lbda, dataset_str = None, None, None
                        try: # block for 1 sec and catch the empty exception
                            fake_images, lbda, dataset_str = self.q.get(block=True, timeout=1)
                        except queue.Empty:
                            continue

                        if fake_images is not None and lbda is not None and dataset_str is not None:
                            async_lbda = rpyc.async_(lbda)
                            self.fid.post(fake_images=rpyc.classic.obtain(fake_images),
                                          lbda=async_lbda, dataset_str=dataset_str)

                    time.sleep(1)

            def post(self, fake_images, lbda, dataset_str):
                """ Posts a set of fake images with a lambda function to operate over FID score

                :param fake_images: the set of numpy fake images
                :param lbda: a lambda function taking 1 param as input
                :param dataset_str: the str name of the dataset
                :returns: None
                :rtype: None

                """
                self.q.put((fake_images, lbda, dataset_str))       # pushes stuff to the queue

        # Singleton to return only one instance
        instance = None

        def __init__(self, normalize=True, force_cpu=False):
            if not FIDService.FID.instance:
                FIDService.FID.instance = FIDService.FID.__FID(
                    normalize=normalize, force_cpu=force_cpu
                )

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
