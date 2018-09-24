# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

This file is used as a module to classify images with Inception3 model in TensorFlow.
Most of the code here comes from TensorFlow authors tutorial on image recognition (see URL below)

https://tensorflow.org/tutorials/image_recognition/

It should be referenced by another Python program that can get a classification from its API


Here is an example of some simple python code that can call the API and classify a folder of images

#############
from inception3_classifier import tf, run_inference_on_image, maybe_download_and_extract
import os, sys

image_dir = './images'


def main(_):
    # downloads the Inception3 training module to ./model folder
    maybe_download_and_extract()
    dlist = os.listdir(image_dir)
    idx = 0
    for f in dlist:
        if 'jpg' in f.lower():
            data = {}
            idx += 1
            # attempt classification
            data = run_inference_on_image(os.path.join(image_dir, f), num_top_predictions=1, verbose=False)
            print("{}. file: {} score: {:4.3f} id: {}".format(idx, f, data[0].get('score'), data[0].get('ident') ))
    print("done")
    sys.exit(0)
            
if __name__ == '__main__':
    tf.app.run(main=main) # tensor flow wraps the app.
###################
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from urllib.request import urlopen
import os


FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None, model_dir='./model'):
        if not label_lookup_path:
            label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph(model_dir='./model'):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, num_top_predictions=5, verbose=True):
    """Runs inference on an image.

    Args:
      image: Image file name.
      num_top_predictions: how many predictions will be returned
      verbose: echos data to the console

    Returns:
      a LIST of its top predictions. Each list item is a dictionary {'score':float, 'ident':string}
      score is a float 0-1.0 which indicates confidence rating. Higher scores indicate higher level of confidence
      ident is a human readable string as to the identity of picture.
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        inf_data = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            if verbose:
                print('%s (score = %.5f)' % (human_string, score))
            inf_data.append({'score':score,'ident':human_string})
        return inf_data

def run_inference_on_url(url, num_top_predictions=5, verbose=True):
    """run_inference_on_url(url, num_top_predictions=5, verbose=True)
    will download the picture (REQUIRED JPG at URL) as temp_image.jpg
    see run_inference_on_image for details"""
    fname = 'temp_image.jpg'
    if download_picture(url, dest_filename=fname, overwrite=True):
        return run_inference_on_image(fname, num_top_predictions=num_top_predictions, verbose=verbose)
    else:
        print("Download of {} failed.".format(url))
        raise ValueError
    

def maybe_download_and_extract(dest_directory='./model'):
    """Download and extract model tar file."""
    #dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    maybe_download_and_extract()
    image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
    run_inference_on_image(image)


def download_picture(url, dest_filename, dest_folder='./', overwrite=False, verbose=False):
    """download_picture(url, dest_filename, dest_folder, overwrite=False)"""
    response = urlopen(url)
    document = response.read()

    outfile = os.path.join(dest_folder, dest_filename)
    if not os.path.isdir(dest_folder):
                # make directory if required
        os.mkdir(dest_folder)

    if os.path.isfile(outfile) and overwrite == False:
        if verbose: print("skipping overwrite of {}".format(outfile))
        return False # don't overwrite existing file
    else:
        if verbose: print("saving file {}".format(outfile))
        with open(outfile,'wb') as fp:
            fp.write(document)
        return True

if __name__ == '__main__':
    print("This module is being used for image classification with Inception3 model.")