#!/usr/bin/env python
"""
Very simple HTTP server in python (Updated for Python 3.7)
Based of: https://gist.github.com/bradmontgomery/2219997

Usage:
    ./server.py -h
    ./server.py -l localhost -p 8000

Send a POST request:
    curl -d "foo=bar&bin=baz" http://localhost:8000
"""

import dill
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler


from fid import AsyncFID



class Server(BaseHTTPRequestHandler):
    def _set_good_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _set_fail_headers(self):
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def get_fid_object(self, dataset_str, root_folder, normalize=True, force_cpu=False):
        if not hasattr(self, 'fid'):
            self.fid = AsyncFID(dataset_str=dataset_str, root_folder=root_folder,
                                normalize=normalize, force_cpu=force_cpu)

        # TODO: logic to add a new dataset
        # self.fid.add_dataset(dataset_str, root_folder)
        return self.fid

    def do_POST(self):
        """ Accepts POST requests and un-dills them.

        :returns: response code
        :rtype: None

        """
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = dill.loads(post_data)

        # sanity checks
        for key in ['dataset_str', 'root_folder', 'fake_images', 'lambda']:
            if key not in post_data:
                print("{} is required for FID calculation, skipping POST".format(key))
                self._set_fail_headers()
                return

        self._set_good_headers()

        # get the FID object
        fid = self.get_fid_object(dataset_str=post_data['dataset_str'],
                                  root_folder=post_data['root_folder'],
                                  normalize=post_data.get('normalize', True),
                                  force_cpu=post_data.get('force_cpu', False))
        fid.post(fake_images=post_data['fake_images'],
                 lbda=post_data['lambda'])


def run(server_class=HTTPServer, handler_class=Server, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FID HTTP server")
    parser.add_argument("-l", "--listen", default="localhost",
        help="Specify the IP address on which the server listens")
    parser.add_argument("-p", "--port", type=int, default=8000,
        help="Specify the port on which the server listens")
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
