# https://github.com/openai/improved-gan/tree/master/inception_score
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

import pathlib
from PIL import Image
from imageio import imread

from tqdm import tqdm, trange

# model.py provides a code to compute Inception Score. Inception model is downloaded automatically. The function to compute score is called "get_inception_score".

MODEL_DIR = './Inception'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  with tf.compat.v1.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in trange(n_batches):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  import tensorboard as tb
  tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
  with open(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.compat.v1.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            # shape = [s.value for s in shape]
            shape = [s for s in shape] #TF 2.x
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if __name__ == '__main__':

  from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("path", type=str, help='Path to the generated images')
  parser.add_argument("--size", type=int, default=0, help='Path to the generated images')
  parser.add_argument('--gpu', type=str, default='0', help='whick GPU to use')    
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  print("To calculate Inception Scores on {}".format(args.path))
  path = pathlib.Path(args.path)
  files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

  def process(x, size=0):
    x = Image.open(str(x)).convert("RGB")
    if size > 0:
      x = x.resize( (size,size) )
    return np.array(x).astype(np.uint8)

  images = [process(fn, args.size) for fn in files]
  # images = [np.array(imread(str(fn))).astype(np.uint8) for fn in files]

  if softmax is None:
    _init_inception()
  
  mean, std = get_inception_score(images, splits=5)

  print(f"Inception score: {mean}+-{std}")
