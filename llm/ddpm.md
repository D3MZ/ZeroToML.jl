This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
diffusion_tf/
  models/
    unet.py
  tpu_utils/
    classifier_metrics_numpy.py
    datasets.py
    simple_eval_worker.py
    tpu_summaries.py
    tpu_utils.py
  diffusion_utils_2.py
  diffusion_utils.py
  nn.py
  utils.py
scripts/
  run_celebahq.py
  run_cifar.py
  run_lsun.py
.gitignore
README.md
requirements.txt
```

# Files

## File: diffusion_tf/models/unet.py
````python
import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib

from .. import nn


def nonlinearity(x):
  return tf.nn.swish(x)


def normalize(x, *, temb, name):
  return tf_contrib.layers.group_norm(x, scope=name)


def upsample(x, *, name, with_conv):
  with tf.variable_scope(name):
    B, H, W, C = x.shape
    x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    assert x.shape == [B, H * 2, W * 2, C]
    if with_conv:
      x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=1)
      assert x.shape == [B, H * 2, W * 2, C]
    return x


def downsample(x, *, name, with_conv):
  with tf.variable_scope(name):
    B, H, W, C = x.shape
    if with_conv:
      x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=2)
    else:
      x = tf.nn.avg_pool(x, 2, 2, 'SAME')
    assert x.shape == [B, H // 2, W // 2, C]
    return x


def resnet_block(x, *, temb, name, out_ch=None, conv_shortcut=False, dropout):
  B, H, W, C = x.shape
  if out_ch is None:
    out_ch = C

  with tf.variable_scope(name):
    h = x

    h = nonlinearity(normalize(h, temb=temb, name='norm1'))
    h = nn.conv2d(h, name='conv1', num_units=out_ch)

    # add in timestep embedding
    h += nn.dense(nonlinearity(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]

    h = nonlinearity(normalize(h, temb=temb, name='norm2'))
    h = tf.nn.dropout(h, rate=dropout)
    h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)

    if C != out_ch:
      if conv_shortcut:
        x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
      else:
        x = nn.nin(x, name='nin_shortcut', num_units=out_ch)

    assert x.shape == h.shape
    print('{}: x={} temb={}'.format(tf.get_default_graph().get_name_scope(), x.shape, temb.shape))
    return x + h


def attn_block(x, *, name, temb):
  B, H, W, C = x.shape
  with tf.variable_scope(name):
    h = normalize(x, temb=temb, name='norm')
    q = nn.nin(h, name='q', num_units=C)
    k = nn.nin(h, name='k', num_units=C)
    v = nn.nin(h, name='v', num_units=C)

    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [B, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [B, H, W, H, W])

    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = nn.nin(h, name='proj_out', num_units=C, init_scale=0.)

    assert h.shape == x.shape
    print(tf.get_default_graph().get_name_scope(), x.shape)
    return x + h


def model(x, *, t, y, name, num_classes, reuse=tf.AUTO_REUSE, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
          attn_resolutions, dropout=0., resamp_with_conv=True):
  B, S, _, _ = x.shape
  assert x.dtype == tf.float32 and x.shape[2] == S
  assert t.dtype in [tf.int32, tf.int64]
  num_resolutions = len(ch_mult)

  assert num_classes == 1 and y is None, 'not supported'
  del y

  with tf.variable_scope(name, reuse=reuse):
    # Timestep embedding
    with tf.variable_scope('temb'):
      temb = nn.get_timestep_embedding(t, ch)
      temb = nn.dense(temb, name='dense0', num_units=ch * 4)
      temb = nn.dense(nonlinearity(temb), name='dense1', num_units=ch * 4)
      assert temb.shape == [B, ch * 4]

    # Downsampling
    hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
    for i_level in range(num_resolutions):
      with tf.variable_scope('down_{}'.format(i_level)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
          h = resnet_block(
            hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
          if h.shape[1] in attn_resolutions:
            h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
          hs.append(h)
        # Downsample
        if i_level != num_resolutions - 1:
          hs.append(downsample(hs[-1], name='downsample', with_conv=resamp_with_conv))

    # Middle
    with tf.variable_scope('mid'):
      h = hs[-1]
      h = resnet_block(h, temb=temb, name='block_1', dropout=dropout)
      h = attn_block(h, name='attn_1'.format(i_block), temb=temb)
      h = resnet_block(h, temb=temb, name='block_2', dropout=dropout)

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      with tf.variable_scope('up_{}'.format(i_level)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks + 1):
          h = resnet_block(tf.concat([h, hs.pop()], axis=-1), name='block_{}'.format(i_block),
                           temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
          if h.shape[1] in attn_resolutions:
            h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
        # Upsample
        if i_level != 0:
          h = upsample(h, name='upsample', with_conv=resamp_with_conv)
    assert not hs

    # End
    h = nonlinearity(normalize(h, temb=temb, name='norm_out'))
    h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
    assert h.shape == x.shape[:3] + [out_ch]
    return h
````

## File: diffusion_tf/tpu_utils/classifier_metrics_numpy.py
````python
"""
Direct NumPy port of tfgan.eval.classifier_metrics
"""

import numpy as np
import scipy.special


def log_softmax(x, axis):
  return x - scipy.special.logsumexp(x, axis=axis, keepdims=True)


def kl_divergence(p, p_logits, q):
  assert len(p.shape) == len(p_logits.shape) == 2
  assert len(q.shape) == 1
  return np.sum(p * (log_softmax(p_logits, axis=1) - np.log(q)[None, :]), axis=1)


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  u, s, vt = np.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = np.where(s < eps, s, np.sqrt(s))
  return u.dot(np.diag(si)).dot(vt)


def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = sqrt_sigma.dot(sigma_v.dot(sqrt_sigma))

  return np.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def classifier_score_from_logits(logits):
  """Classifier score for evaluating a generative model from logits.

  This method computes the classifier score for a set of logits. This can be
  used independently of the classifier_score() method, especially in the case
  of using large batches during evaluation where we would like precompute all
  of the logits before computing the classifier score.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates:

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  Args:
    logits: Precomputed 2D tensor of logits that will be used to compute the
      classifier score.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `logits`.
  """
  assert len(logits.shape) == 2

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != np.float64:
    logits = logits.astype(np.float64)

  p = scipy.special.softmax(logits, axis=1)
  q = np.mean(p, axis=0)
  kl = kl_divergence(p, logits, q)
  assert len(kl.shape) == 1
  log_score = np.mean(kl)
  final_score = np.exp(log_score)

  if logits_dtype != np.float64:
    final_score = final_score.astype(logits_dtype)

  return final_score


def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
  """Classifier distance for evaluating a generative model.

  This methods computes the Frechet classifier distance from activations of
  real images and generated images. This can be used independently of the
  frechet_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like precompute all of the
  activations before computing the classifier distance.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

                |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.

  """
  assert len(real_activations.shape) == len(generated_activations.shape) == 2

  activations_dtype = real_activations.dtype
  if activations_dtype != np.float64:
    real_activations = real_activations.astype(np.float64)
    generated_activations = generated_activations.astype(np.float64)

  # Compute mean and covariance matrices of activations.
  m = np.mean(real_activations, 0)
  m_w = np.mean(generated_activations, 0)
  num_examples_real = float(real_activations.shape[0])
  num_examples_generated = float(generated_activations.shape[0])

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  real_centered = real_activations - m
  sigma = real_centered.T.dot(real_centered) / (num_examples_real - 1)

  gen_centered = generated_activations - m_w
  sigma_w = gen_centered.T.dot(gen_centered) / (num_examples_generated - 1)

  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = np.sum(np.square(m - m_w))  # Equivalent to L2 but more stable.
  fid = trace + mean
  if activations_dtype != np.float64:
    fid = fid.astype(activations_dtype)

  return fid


def test_all():
  """
  Test against tfgan.eval.classifier_metrics
  """

  import tensorflow.compat.v1 as tf
  import tensorflow_gan as tfgan

  rand = np.random.RandomState(1234)
  logits = rand.randn(64, 1008)
  asdf1, asdf2 = rand.randn(64, 2048), rand.rand(256, 2048)
  with tf.Session() as sess:
    assert np.allclose(
      sess.run(tfgan.eval.classifier_score_from_logits(tf.convert_to_tensor(logits))),
      classifier_score_from_logits(logits))
    assert np.allclose(
      sess.run(tfgan.eval.frechet_classifier_distance_from_activations(
        tf.convert_to_tensor(asdf1), tf.convert_to_tensor(asdf2))),
      frechet_classifier_distance_from_activations(asdf1, asdf2))
  print('all ok')


if __name__ == '__main__':
  test_all()
````

## File: diffusion_tf/tpu_utils/datasets.py
````python
"""Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def pack(image, label):
  label = tf.cast(label, tf.int32)
  return {'image': image, 'label': label}


class SimpleDataset:
  DATASET_NAMES = ('cifar10', 'celebahq256')

  def __init__(self, name, tfds_data_dir):
    self._name = name
    self._data_dir = tfds_data_dir
    self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
    self._img_shape = [self._img_size, self._img_size, 3]
    self._tfds_name = {
      'cifar10': 'cifar10:3.0.0',
      'celebahq256': 'celeb_a_hq/256:2.0.0',
    }[name]
    self.num_train_examples, self.num_eval_examples = {
      'cifar10': (50000, 10000),
      'celebahq256': (30000, 0),
    }[name]
    self.num_classes = 1  # unconditional
    self.eval_split_name = {
      'cifar10': 'test',
      'celebahq256': None,
    }[name]

  @property
  def image_shape(self):
    """Returns a tuple with the image shape."""
    return tuple(self._img_shape)

  def _proc_and_batch(self, ds, batch_size):
    def _process_data(x_):
      img_ = tf.cast(x_['image'], tf.int32)
      img_.set_shape(self._img_shape)
      return pack(image=img_, label=tf.constant(0, dtype=tf.int32))

    ds = ds.map(_process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=True, data_dir=self._data_dir)
    ds = ds.repeat()
    ds = ds.shuffle(50000)
    return self._proc_and_batch(ds, params['batch_size'])

  def train_one_pass_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=False, data_dir=self._data_dir)
    return self._proc_and_batch(ds, params['batch_size'])

  def eval_input_fn(self, params):
    if self.eval_split_name is None:
      return None
    ds = tfds.load(self._tfds_name, split=self.eval_split_name, shuffle_files=False, data_dir=self._data_dir)
    return self._proc_and_batch(ds, params['batch_size'])


class LsunDataset:
  def __init__(self,
    tfr_file,            # Path to tfrecord file.
    resolution=256,      # Dataset resolution.
    max_images=None,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file           = tfr_file
    self.dtype              = 'int32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 1         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]

  def _train_input_fn(self, params, one_pass: bool):
    # Build TF expressions.
    dset = tf.data.TFRecordDataset(self.tfr_file, compression_type='', buffer_size=self.buffer_mb<<20)
    if self.max_images is not None:
      dset = dset.take(self.max_images)
    if not one_pass:
      dset = dset.repeat()
    dset = dset.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and prefetch
    dset = dset.shuffle(50000)
    dset = dset.batch(params['batch_size'], drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset

  def train_input_fn(self, params):
    return self._train_input_fn(params, one_pass=False)

  def train_one_pass_input_fn(self, params):
    return self._train_input_fn(params, one_pass=True)

  def eval_input_fn(self, params):
    return None

  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf(self, record):
    features = tf.parse_single_example(record, features={
      'shape': tf.FixedLenFeature([3], tf.int64),
      'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    data = tf.cast(data, tf.int32)
    data = tf.reshape(data, features['shape'])
    data = tf.transpose(data, [1, 2, 0])  # CHW -> HWC
    data.set_shape(self.image_shape)
    return pack(image=data, label=tf.constant(0, dtype=tf.int32))


DATASETS = {
  "cifar10": functools.partial(SimpleDataset, name="cifar10"),
  "celebahq256": functools.partial(SimpleDataset, name="celebahq256"),
  "lsun": LsunDataset,
}


def get_dataset(name, *, tfds_data_dir=None, tfr_file=None, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  kwargs = {}

  if name == 'lsun':
    # LsunDataset takes the path to the tf record, not a directory
    assert tfr_file is not None
    kwargs['tfr_file'] = tfr_file
  else:
    kwargs['tfds_data_dir'] = tfds_data_dir

  if name not in ['lsun', *SimpleDataset.DATASET_NAMES]:
    kwargs['seed'] = seed

  return DATASETS[name](**kwargs)
````

## File: diffusion_tf/tpu_utils/simple_eval_worker.py
````python
"""
"One-shot" evaluation worker (i.e. run something once, not in a loop over the course of training)

- Computes log prob
- Generates samples progressively
"""

import os
import pickle
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import trange

from .tpu_utils import Model, make_ema, distributed, normalize_data
from .. import utils


def _make_ds_iterator(strategy, ds):
  return strategy.experimental_distribute_dataset(ds).make_initializable_iterator()


class SimpleEvalWorker:
  def __init__(self, tpu_name, model_constructor, total_bs, dataset):
    tf.logging.set_verbosity(tf.logging.INFO)

    self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.tpu.experimental.initialize_tpu_system(self.resolver)
    self.strategy = tf.distribute.experimental.TPUStrategy(self.resolver)

    self.num_cores = self.strategy.num_replicas_in_sync
    assert total_bs % self.num_cores == 0
    self.total_bs = total_bs
    self.local_bs = total_bs // self.num_cores
    print('num cores: {}'.format(self.num_cores))
    print('total batch size: {}'.format(self.total_bs))
    print('local batch size: {}'.format(self.local_bs))
    self.dataset = dataset

    # TPU context
    with self.strategy.scope():
      # Dataset iterators
      self.train_ds_iterator = _make_ds_iterator(
        self.strategy, dataset.train_one_pass_input_fn(params={'batch_size': total_bs}))
      self.eval_ds_iterator = _make_ds_iterator(
        self.strategy, dataset.eval_input_fn(params={'batch_size': total_bs}))

      img_batch_shape = self.train_ds_iterator.output_shapes['image'].as_list()
      assert img_batch_shape[0] == self.local_bs

      # Model
      self.model = model_constructor()
      assert isinstance(self.model, Model)

      # Eval/samples graphs
      print('===== SAMPLES =====')
      self.samples_outputs = self._make_progressive_sampling_graph(img_shape=img_batch_shape[1:])

      # Model with EMA parameters
      print('===== EMA =====')
      self.global_step = tf.train.get_or_create_global_step()
      ema, _ = make_ema(global_step=self.global_step, ema_decay=1e-10, trainable_variables=tf.trainable_variables())

      # EMA versions of the above
      with utils.ema_scope(ema):
        print('===== EMA SAMPLES =====')
        self.ema_samples_outputs = self._make_progressive_sampling_graph(img_shape=img_batch_shape[1:])
        print('===== EMA BPD =====')
        self.bpd_train = self._make_bpd_graph(self.train_ds_iterator)
        self.bpd_eval = self._make_bpd_graph(self.eval_ds_iterator)

  def _make_progressive_sampling_graph(self, img_shape):
    return distributed(
      lambda x_: self.model.progressive_samples_fn(
        x_, tf.random_uniform([self.local_bs], 0, self.dataset.num_classes, dtype=tf.int32)),
      args=(tf.fill([self.local_bs, *img_shape], value=np.nan),),
      reduction='concat', strategy=self.strategy)

  def _make_bpd_graph(self, ds_iterator):
    return distributed(
      lambda x_: self.model.bpd_fn(normalize_data(tf.cast(x_['image'], tf.float32)), x_['label']),
      args=(next(ds_iterator),), reduction='concat', strategy=self.strategy)

  def init_all_iterators(self, sess):
    sess.run([self.train_ds_iterator.initializer, self.eval_ds_iterator.initializer])

  def dump_progressive_samples(self, sess, curr_step, samples_dir, ema: bool, num_samples=50000, batches_per_flush=20):
    if not tf.gfile.IsDirectory(samples_dir):
      tf.gfile.MakeDirs(samples_dir)

    batch_cache, num_flushes_so_far = [], 0

    def _write_batch_cache():
      nonlocal batch_cache, num_flushes_so_far
      # concat all the batches
      assert all(set(b.keys()) == set(self.samples_outputs.keys()) for b in batch_cache)
      concatenated = {
        k: np.concatenate([b[k].astype(np.float32) for b in batch_cache], axis=0)
        for k in self.samples_outputs.keys()
      }
      assert len(set(len(v) for v in concatenated.values())) == 1
      # write the file
      filename = os.path.join(
        samples_dir, 'samples_xstartpred_ema{}_step{:09d}_part{:06d}.pkl'.format(
          int(ema), curr_step, num_flushes_so_far))
      assert not tf.io.gfile.exists(filename), 'samples file already exists: {}'.format(filename)
      print('writing samples batch to:', filename)
      with tf.io.gfile.GFile(filename, 'wb') as f:
        f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))
      print('done writing samples batch')
      num_flushes_so_far += 1
      batch_cache = []

    num_gen_batches = int(np.ceil(num_samples / self.total_bs))
    print('generating {} samples ({} batches)...'.format(num_samples, num_gen_batches))
    self.init_all_iterators(sess)
    for i_batch in trange(num_gen_batches, desc='sampling'):
      batch_cache.append(sess.run(self.ema_samples_outputs if ema else self.samples_outputs))
      if i_batch != 0 and i_batch % batches_per_flush == 0:
        _write_batch_cache()
    if batch_cache:
      _write_batch_cache()

  def dump_bpd(self, sess, curr_step, output_dir, train: bool, ema: bool):
    assert ema
    if not tf.gfile.IsDirectory(output_dir):
      tf.gfile.MakeDirs(output_dir)
    filename = os.path.join(
      output_dir, 'bpd_{}_ema{}_step{:09d}.pkl'.format('train' if train else 'eval', int(ema), curr_step))
    assert not tf.io.gfile.exists(filename), 'bpd file already exists: {}'.format(filename)
    print('will write bpd data to:', filename)

    batches = []
    tf_op = self.bpd_train if train else self.bpd_eval
    self.init_all_iterators(sess)
    last_print_time = time.time()
    while True:
      try:
        batches.append(sess.run(tf_op))
        if time.time() - last_print_time > 30:
          print('num batches so far: {} ({:.2f} sec)'.format(len(batches), time.time() - last_print_time))
          last_print_time = time.time()
      except tf.errors.OutOfRangeError:
        break

    assert all(set(b.keys()) == set(tf_op.keys()) for b in batches)
    concatenated = {
      k: np.concatenate([b[k].astype(np.float32) for b in batches], axis=0)
      for k in tf_op.keys()
    }
    num_samples = len(list(concatenated.values())[0])
    assert all(len(v) == num_samples for v in concatenated.values())
    print('evaluated on {} examples'.format(num_samples))

    print('writing results to:', filename)
    with tf.io.gfile.GFile(filename, 'wb') as f:
      f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))
    print('done writing results')

  def run(self, mode: str, logdir: str, load_ckpt: str):
    """
    Main entry point.

    :param mode: what to do
    :param logdir: model directory for the checkpoint to load
    :param load_ckpt: the name of the checkpoint, e.g. "model.ckpt-1000000"
    """

    # Input checkpoint: load_ckpt should be of the form: model.ckpt-1000000
    ckpt = os.path.join(logdir, load_ckpt)
    assert tf.io.gfile.exists(ckpt + '.index')

    # Output dir
    output_dir = os.path.join(logdir, 'simple_eval')
    print('Writing output to: {}'.format(output_dir))

    # Make the session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    cluster_spec = self.resolver.cluster_spec()
    if cluster_spec:
      config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    print('making session...')
    with tf.Session(target=self.resolver.master(), config=config) as sess:

      print('initializing global variables')
      sess.run(tf.global_variables_initializer())

      # Checkpoint loading
      print('making saver')
      saver = tf.train.Saver()
      saver.restore(sess, ckpt)
      global_step_val = sess.run(self.global_step)
      print('restored global step: {}'.format(global_step_val))

      if mode in ['bpd_train', 'bpd_eval']:
        self.dump_bpd(
          sess, curr_step=global_step_val, output_dir=os.path.join(output_dir, 'bpd'), ema=True,
          train=mode == 'bpd_train')
      elif mode == 'progressive_samples':
        return self.dump_progressive_samples(
          sess, curr_step=global_step_val, samples_dir=os.path.join(output_dir, 'progressive_samples'), ema=True)
      else:
        raise NotImplementedError(mode)
````

## File: diffusion_tf/tpu_utils/tpu_summaries.py
````python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import tensorflow as tf


summary = tf.contrib.summary  # TensorFlow Summary API v2.


TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn")


class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, save_summary_steps=250):
    self._log_dir = log_dir
    self._entries = []
    # While False no summary entries will be added. On TPU we unroll the graph
    # and don't want to add multiple summaries per step.
    self.record = True
    self._save_summary_steps = save_summary_steps

  def image(self, name, tensor, reduce_fn):
    """Add a summary for images. Tensor must be of 4-D tensor."""
    if not self.record:
      return
    self._entries.append(
        TpuSummaryEntry(summary.image, name, tensor, reduce_fn))

  def scalar(self, name, tensor, reduce_fn=tf.math.reduce_mean):
    """Add a summary for a scalar tensor."""
    if not self.record:
      return
    tensor = tf.convert_to_tensor(tensor)
    if tensor.shape.ndims == 0:
      tensor = tf.expand_dims(tensor, 0)
    self._entries.append(
        TpuSummaryEntry(summary.scalar, name, tensor, reduce_fn))

  def get_host_call(self):
    """Returns the tuple (host_call_fn, host_call_args) for TPUEstimatorSpec."""
    # All host_call_args must be tensors with batch dimension.
    # All tensors are streamed to the host machine (mind the band width).
    global_step = tf.train.get_or_create_global_step()
    host_call_args = [tf.expand_dims(global_step, 0)]
    host_call_args.extend([e.tensor for e in self._entries])
    logging.info("host_call_args: %s", host_call_args)
    return (self._host_call_fn, host_call_args)

  def _host_call_fn(self, step, *args):
    """Function that will run on the host machine."""
    # Host call receives values from all tensor cores (concatenate on the
    # batch dimension). Step is the same for all cores.
    step = step[0]
    logging.info("host_call_fn: args=%s", args)
    with summary.create_file_writer(self._log_dir).as_default():
      with summary.record_summaries_every_n_global_steps(
          self._save_summary_steps, step):
        for i, e in enumerate(self._entries):
          value = e.reduce_fn(args[i])
          e.summary_fn(e.name, value, step=step)
        return summary.all_summary_ops()
````

## File: diffusion_tf/tpu_utils/tpu_utils.py
````python
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.tpu import tpu_function
from tqdm import trange

from . import classifier_metrics_numpy
from .tpu_summaries import TpuSummaries
from .. import utils


# ========== TPU utilities ==========

def num_tpu_replicas():
  return tpu_function.get_tpu_context().number_of_shards


def get_tpu_replica_id():
  with tf.control_dependencies(None):
    return tpu_ops.tpu_replicated_input(list(range(num_tpu_replicas())))


def distributed(fn, *, args, reduction, strategy):
  """
  Sharded computation followed by concat/mean for TPUStrategy.
  """
  out = strategy.experimental_run_v2(fn, args=args)
  if reduction == 'mean':
    return tf.nest.map_structure(lambda x: tf.reduce_mean(strategy.reduce('mean', x)), out)
  elif reduction == 'concat':
    return tf.nest.map_structure(lambda x: tf.concat(strategy.experimental_local_results(x), axis=0), out)
  else:
    raise NotImplementedError(reduction)


# ========== Inception utilities ==========

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05_v4.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score_tpu.pb'
INCEPTION_GRAPH_DEF = tfgan.eval.get_graph_def_from_url_tarball(
  INCEPTION_URL, INCEPTION_FROZEN_GRAPH, os.path.basename(INCEPTION_URL))


def run_inception(images):
  assert images.dtype == tf.float32  # images should be in [-1, 1]
  out = tfgan.eval.run_inception(
    images,
    graph_def=INCEPTION_GRAPH_DEF,
    default_graph_def_fn=None,
    output_tensor=['pool_3:0', 'logits:0']
  )
  return {'pool_3': out[0], 'logits': out[1]}


# ========== Training ==========

normalize_data = lambda x_: x_ / 127.5 - 1.
unnormalize_data = lambda x_: (x_ + 1.) * 127.5


class Model:
  # All images (inputs and outputs) should be normalized to [-1, 1]
  def train_fn(self, x, y) -> dict:
    raise NotImplementedError

  def samples_fn(self, dummy_x, y) -> dict:
    raise NotImplementedError

  def sample_and_run_inception(self, dummy_x, y, clip_samples=True):
    samples_dict = self.samples_fn(dummy_x, y)
    assert isinstance(samples_dict, dict)
    return {
      k: run_inception(tfgan.eval.preprocess_image(unnormalize_data(
        tf.clip_by_value(v, -1., 1.) if clip_samples else v)))
      for (k, v) in samples_dict.items()
    }

  def bpd_fn(self, x, y) -> dict:
    return None


def make_ema(global_step, ema_decay, trainable_variables):
  ema = tf.train.ExponentialMovingAverage(decay=tf.where(tf.less(global_step, 1), 1e-10, ema_decay))
  ema_op = ema.apply(trainable_variables)
  return ema, ema_op


def load_train_kwargs(model_dir):
  with tf.io.gfile.GFile(os.path.join(model_dir, 'kwargs.json'), 'r') as f:
    kwargs = json.loads(f.read())
  return kwargs


def run_training(
    *, model_constructor, train_input_fn, total_bs,
    optimizer, lr, warmup, grad_clip, ema_decay=0.9999,
    tpu=None, zone=None, project=None,
    log_dir, exp_name, dump_kwargs=None,
    date_str=None, iterations_per_loop=1000, keep_checkpoint_max=2, max_steps=int(1e10),
    warm_start_from=None
):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create checkpoint directory
  model_dir = os.path.join(
    log_dir,
    datetime.now().strftime('%Y-%m-%d') if date_str is None else date_str,
    exp_name
  )
  print('model dir:', model_dir)
  if tf.io.gfile.exists(model_dir):
    print('model dir already exists: {}'.format(model_dir))
    if input('continue training? [y/n] ') != 'y':
      print('aborting')
      return

  # Save kwargs in json format
  if dump_kwargs is not None:
    with tf.io.gfile.GFile(os.path.join(model_dir, 'kwargs.json'), 'w') as f:
      f.write(json.dumps(dump_kwargs, indent=2, sort_keys=True) + '\n')

  # model_fn for TPUEstimator
  def model_fn(features, params, mode):
    local_bs = params['batch_size']
    print('Global batch size: {}, local batch size: {}'.format(total_bs, local_bs))
    assert total_bs == num_tpu_replicas() * local_bs

    assert mode == tf.estimator.ModeKeys.TRAIN, 'only TRAIN mode supported'
    assert features['image'].shape[0] == local_bs
    assert features['label'].shape == [local_bs] and features['label'].dtype == tf.int32
    # assert labels.dtype == features['label'].dtype and labels.shape == features['label'].shape

    del params

    ##########

    # create model
    model = model_constructor()
    assert isinstance(model, Model)

    # training loss
    train_info_dict = model.train_fn(normalize_data(tf.cast(features['image'], tf.float32)), features['label'])
    loss = train_info_dict['loss']
    assert loss.shape == []

    # train op
    trainable_variables = tf.trainable_variables()
    print('num params: {:,}'.format(sum(int(np.prod(p.shape.as_list())) for p in trainable_variables)))
    global_step = tf.train.get_or_create_global_step()
    warmed_up_lr = utils.get_warmed_up_lr(max_lr=lr, warmup=warmup, global_step=global_step)
    train_op, gnorm = utils.make_optimizer(
      loss=loss,
      trainable_variables=trainable_variables,
      global_step=global_step,
      lr=warmed_up_lr,
      optimizer=optimizer,
      grad_clip=grad_clip / float(num_tpu_replicas()),
      tpu=True
    )

    # ema
    ema, ema_op = make_ema(global_step=global_step, ema_decay=ema_decay, trainable_variables=trainable_variables)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(ema_op)

    # summary
    tpu_summary = TpuSummaries(model_dir, save_summary_steps=100)
    tpu_summary.scalar('train/loss', loss)
    tpu_summary.scalar('train/gnorm', gnorm)
    tpu_summary.scalar('train/pnorm', utils.rms(trainable_variables))
    tpu_summary.scalar('train/lr', warmed_up_lr)
    return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode, host_call=tpu_summary.get_host_call(), loss=loss, train_op=train_op)

  # Set up Estimator and train
  print("warm_start_from:", warm_start_from)
  estimator = tf.estimator.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=True,
    train_batch_size=total_bs,
    eval_batch_size=total_bs,
    config=tf.estimator.tpu.RunConfig(
      cluster=tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu, zone=zone, project=project),
      model_dir=model_dir,
      session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.estimator.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=None,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
      ),
      save_checkpoints_secs=1600,  # 30 minutes
      keep_checkpoint_max=keep_checkpoint_max
    ),
    warm_start_from=warm_start_from
  )
  estimator.train(input_fn=train_input_fn, max_steps=max_steps)


# ========== Evaluation / sampling ==========


class InceptionFeatures:
  """
  Compute and store Inception features for a dataset
  """

  def __init__(self, dataset, strategy, limit_dataset_size=0):
    # distributed dataset iterator
    if limit_dataset_size > 0:
      dataset = dataset.take(limit_dataset_size)
    self.ds_iterator = strategy.experimental_distribute_dataset(dataset).make_initializable_iterator()

    # inception network on the dataset
    self.inception_real = distributed(
      lambda x_: run_inception(tfgan.eval.preprocess_image(x_['image'])),
      args=(next(self.ds_iterator),), reduction='concat', strategy=strategy)

    self.cached_inception_real = None  # cached inception features
    self.real_inception_score = None  # saved inception scores for the dataset

  def get(self, sess):
    # On the first invocation, compute Inception activations for the eval dataset
    if self.cached_inception_real is None:
      print('computing inception features on the eval set...')
      sess.run(self.ds_iterator.initializer)  # reset the eval dataset iterator
      inception_real_batches, tstart = [], time.time()
      while True:
        try:
          inception_real_batches.append(sess.run(self.inception_real))
        except tf.errors.OutOfRangeError:
          break
      self.cached_inception_real = {
        feat_key: np.concatenate([batch[feat_key] for batch in inception_real_batches], axis=0).astype(np.float64)
        for feat_key in ['pool_3', 'logits']
      }
      print('cached eval inception tensors: logits: {}, pool_3: {} (time: {})'.format(
        self.cached_inception_real['logits'].shape, self.cached_inception_real['pool_3'].shape,
        time.time() - tstart))

      self.real_inception_score = float(
        classifier_metrics_numpy.classifier_score_from_logits(self.cached_inception_real['logits']))
      del self.cached_inception_real['logits']  # save memory
    print('real inception score', self.real_inception_score)

    return self.cached_inception_real, self.real_inception_score


class EvalWorker:
  def __init__(self, tpu_name, model_constructor, total_bs, dataset, inception_bs=8, num_inception_samples=1024, limit_dataset_size=0):

    self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.tpu.experimental.initialize_tpu_system(self.resolver)
    self.strategy = tf.distribute.experimental.TPUStrategy(self.resolver)

    self.num_cores = self.strategy.num_replicas_in_sync
    assert total_bs % self.num_cores == 0
    self.total_bs = total_bs
    self.local_bs = total_bs // self.num_cores
    print('num cores: {}'.format(self.num_cores))
    print('total batch size: {}'.format(self.total_bs))
    print('local batch size: {}'.format(self.local_bs))
    self.num_inception_samples = num_inception_samples
    assert inception_bs % self.num_cores == 0
    self.inception_bs = inception_bs
    self.inception_local_bs = inception_bs // self.num_cores
    self.dataset = dataset
    assert dataset.num_classes == 1, 'not supported'

    # TPU context
    with self.strategy.scope():
      # Inception network on real data
      print('===== INCEPTION =====')
      # Eval dataset iterator (this is the training set without repeat & shuffling)
      self.inception_real_train = InceptionFeatures(
        dataset=dataset.train_one_pass_input_fn(params={'batch_size': total_bs}), strategy=self.strategy, limit_dataset_size=limit_dataset_size // total_bs)
      # Val dataset, if it exists
      val_ds = dataset.eval_input_fn(params={'batch_size': total_bs})
      self.inception_real_val = None if val_ds is None else InceptionFeatures(dataset=val_ds, strategy=self.strategy, limit_dataset_size=limit_dataset_size // total_bs)

      img_batch_shape = self.inception_real_train.ds_iterator.output_shapes['image'].as_list()
      assert img_batch_shape[0] == self.local_bs

      # Model
      self.model = model_constructor()
      assert isinstance(self.model, Model)

      # Eval/samples graphs
      print('===== SAMPLES =====')
      self.samples_outputs, self.samples_inception = self._make_sampling_graph(
        img_shape=img_batch_shape[1:], with_inception=True)

      # Model with EMA parameters
      self.global_step = tf.train.get_or_create_global_step()
      print('===== EMA =====')
      ema, _ = make_ema(global_step=self.global_step, ema_decay=1e-10, trainable_variables=tf.trainable_variables())

      # EMA versions of the above
      with utils.ema_scope(ema):
        print('===== EMA SAMPLES =====')
        self.ema_samples_outputs, self.ema_samples_inception = self._make_sampling_graph(
          img_shape=img_batch_shape[1:], with_inception=True)

  def _make_sampling_graph(self, img_shape, with_inception):

    def _make_inputs(total_bs, local_bs):
      # Dummy inputs to feed to samplers
      input_x = tf.fill([local_bs, *img_shape], value=np.nan)
      input_y = tf.random_uniform([local_bs], 0, self.dataset.num_classes, dtype=tf.int32)
      return input_x, input_y

    # Samples
    samples_outputs = distributed(
      self.model.samples_fn,
      args=_make_inputs(self.total_bs, self.local_bs),
      reduction='concat', strategy=self.strategy)
    if not with_inception:
      return samples_outputs

    # Inception activations of samples
    samples_inception = distributed(
      self.model.sample_and_run_inception,
      args=_make_inputs(self.inception_bs, self.inception_local_bs),
      reduction='concat', strategy=self.strategy)
    return samples_outputs, samples_inception

  def _run_sampling(self, sess, ema: bool):
    out = {}
    print('sampling...')
    tstart = time.time()
    samples = sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
    print('sampling done in {} sec'.format(time.time() - tstart))
    for k, v in samples.items():
      out['samples/{}'.format(k)] = v
    return out

  def _run_metrics(self, sess, ema: bool):
    print('computing sample quality metrics...')
    metrics = {}

    # Get Inception activations on the real dataset
    cached_inception_real_train, metrics['real_inception_score_train'] = self.inception_real_train.get(sess)
    if self.inception_real_val is not None:
      cached_inception_real_val, metrics['real_inception_score'] = self.inception_real_val.get(sess)
    else:
      cached_inception_real_val = None

    # Generate lots of samples
    num_inception_gen_batches = int(np.ceil(self.num_inception_samples / self.inception_bs))
    print('generating {} samples and inception features ({} batches)...'.format(
      self.num_inception_samples, num_inception_gen_batches))
    inception_gen_batches = [
      sess.run(self.ema_samples_inception if ema else self.samples_inception)
      for _ in trange(num_inception_gen_batches, desc='sampling inception batch')
    ]

    # Compute FID and Inception score
    assert set(self.samples_outputs.keys()) == set(inception_gen_batches[0].keys())
    for samples_key in self.samples_outputs.keys():
      # concat features from all batches into a single array
      inception_gen = {
        feat_key: np.concatenate(
          [batch[samples_key][feat_key] for batch in inception_gen_batches], axis=0
        )[:self.num_inception_samples].astype(np.float64)
        for feat_key in ['pool_3', 'logits']
      }
      assert all(v.shape[0] == self.num_inception_samples for v in inception_gen.values())

      # Inception score
      metrics['{}/inception{}'.format(samples_key, self.num_inception_samples)] = float(
        classifier_metrics_numpy.classifier_score_from_logits(inception_gen['logits']))

      # FID vs training set
      metrics['{}/trainfid{}'.format(samples_key, self.num_inception_samples)] = float(
        classifier_metrics_numpy.frechet_classifier_distance_from_activations(
          cached_inception_real_train['pool_3'], inception_gen['pool_3']))

      # FID vs val set
      if cached_inception_real_val is not None:
        metrics['{}/fid{}'.format(samples_key, self.num_inception_samples)] = float(
          classifier_metrics_numpy.frechet_classifier_distance_from_activations(
            cached_inception_real_val['pool_3'], inception_gen['pool_3']))

    return metrics

  def _write_eval_and_samples(self, sess, log: utils.SummaryWriter, curr_step, prefix, ema: bool):
    # Samples
    for k, v in self._run_sampling(sess, ema=ema).items():
      assert len(v.shape) == 4 and v.shape[0] == self.total_bs
      log.images('{}/{}'.format(prefix, k), np.clip(unnormalize_data(v), 0, 255).astype('uint8'), step=curr_step)
    log.flush()

    # Metrics
    metrics = self._run_metrics(sess, ema=ema)
    print('metrics:', json.dumps(metrics, indent=2, sort_keys=True))
    for k, v in metrics.items():
      log.scalar('{}/{}'.format(prefix, k), v, step=curr_step)
    log.flush()

  def _dump_samples(self, sess, curr_step, samples_dir, ema: bool, num_samples=50000):
    print('will dump samples to', samples_dir)
    if not tf.gfile.IsDirectory(samples_dir):
      tf.gfile.MakeDirs(samples_dir)
    filename = os.path.join(
      samples_dir, 'samples_ema{}_step{:09d}.pkl'.format(int(ema), curr_step))
    assert not tf.io.gfile.exists(filename), 'samples file already exists: {}'.format(filename)

    num_gen_batches = int(np.ceil(num_samples / self.total_bs))
    print('generating {} samples ({} batches)...'.format(num_samples, num_gen_batches))

    # gen_batches = [
    #   sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
    #   for _ in trange(num_gen_batches, desc='sampling')
    # ]
    # assert all(set(b.keys()) == set(self.samples_outputs.keys()) for b in gen_batches)
    # concatenated = {
    #   k: np.concatenate([b[k].astype(np.float32) for b in gen_batches], axis=0)[:num_samples]
    #   for k in self.samples_outputs.keys()
    # }
    # assert all(len(v) == num_samples for v in concatenated.values())
    #
    # print('writing samples to:', filename)
    # with tf.io.gfile.GFile(filename, 'wb') as f:
    #   f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))

    for i in trange(num_gen_batches, desc='sampling'):
        b = sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
        assert set(b.keys()) == set(self.samples_outputs.keys())
        b = {
          k: b[k].astype(np.float32) for k in self.samples_outputs.keys()
        }
        #assert all(len(v) == num_samples for v in concatenated.values())

        filename_i = "{}.batch{:05d}".format(filename, i)
        print('writing samples for batch', i, 'to:', filename_i)
        with tf.io.gfile.GFile(filename_i, 'wb') as f:
            f.write(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
    print('done writing samples')

  def run(self, logdir, once: bool, skip_non_ema_pass=True, dump_samples_only=False, load_ckpt=None, samples_dir=None, seed=0):
    """Runs the eval/sampling worker loop.
    Args:
      logdir: directory to read checkpoints from
      once: if True, writes results to a temporary directory (not to logdir),
        and exits after evaluating one checkpoint.
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    # Are we evaluating a single checkpoint or looping on the latest?
    if load_ckpt is not None:
      # load_ckpt should be of the form: model.ckpt-1000000
      assert tf.io.gfile.exists(os.path.join(logdir, load_ckpt) + '.index')
      ckpt_iterator = [os.path.join(logdir, load_ckpt)]  # load this one checkpoint only
    else:
      ckpt_iterator = tf.train.checkpoints_iterator(logdir)  # wait for checkpoints to come in
    assert tf.io.gfile.isdir(logdir), 'expected {} to be a directory'.format(logdir)

    # Set up eval SummaryWriter
    if once:
      eval_logdir = os.path.join(logdir, 'eval_once_{}'.format(time.time()))
    else:
      eval_logdir = os.path.join(logdir, 'eval')
    print('Writing eval data to: {}'.format(eval_logdir))
    eval_log = utils.SummaryWriter(eval_logdir, write_graph=False)

    # Make the session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    cluster_spec = self.resolver.cluster_spec()
    if cluster_spec:
      config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    print('making session...')
    with tf.Session(target=self.resolver.master(), config=config) as sess:

      print('initializing global variables')
      sess.run(tf.global_variables_initializer())

      # Checkpoint loading
      print('making saver')
      saver = tf.train.Saver()

      for ckpt in ckpt_iterator:
        # Restore params
        saver.restore(sess, ckpt)
        global_step_val = sess.run(self.global_step)
        print('restored global step: {}'.format(global_step_val))

        print('seeding')
        utils.seed_all(seed)

        print('ema pass')
        if dump_samples_only:
          if not samples_dir:
            samples_dir = os.path.join(eval_logdir, '{}_samples{}'.format(type(self.dataset).__name__, global_step_val))
          self._dump_samples(
            sess, curr_step=global_step_val, samples_dir=samples_dir, ema=True)
        else:
          self._write_eval_and_samples(sess, log=eval_log, curr_step=global_step_val, prefix='eval_ema', ema=True)

        if not skip_non_ema_pass:
          print('non-ema pass')
          if dump_samples_only:
            self._dump_samples(
              sess, curr_step=global_step_val, samples_dir=os.path.join(eval_logdir, 'samples'), ema=False)
          else:
            self._write_eval_and_samples(sess, log=eval_log, curr_step=global_step_val, prefix='eval', ema=False)

        if once:
          break
````

## File: diffusion_tf/diffusion_utils_2.py
````python
import numpy as np
import tensorflow.compat.v1 as tf

from . import nn
from . import utils


def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  KL divergence between normal distributions parameterized by mean and log-variance.
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
                + tf.squared_difference(mean1, mean2) * tf.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


class GaussianDiffusion2:
  """
  Contains utilities for the diffusion model.

  Arguments:
  - what the network predicts (x_{t-1}, x_0, or epsilon)
  - which loss function (kl or unweighted MSE)
  - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
  - what type of decoder, and how to weight its loss? is its variance learned too?
  """

  def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
    self.model_mean_type = model_mean_type  # xprev, xstart, eps
    self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
    self.loss_type = loss_type  # kl, mse

    assert isinstance(betas, np.ndarray)
    self.betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
    assert (betas > 0).all() and (betas <= 1).all()
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    alphas = 1. - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
    assert self.alphas_cumprod_prev.shape == (timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
    self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    out = tf.gather(tf.convert_to_tensor(a, dtype=tf.float32), t)
    assert out.shape == [bs]
    return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape)
    assert noise.shape == x_start.shape
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    posterior_mean = (
        self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
            x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
    B, H, W, C = x.shape
    assert t.shape == [B]
    model_output = denoise_fn(x, t)

    # Learned or fixed variance?
    if self.model_var_type == 'learned':
      assert model_output.shape == [B, H, W, C * 2]
      model_output, model_log_variance = tf.split(model_output, 2, axis=-1)
      model_variance = tf.exp(model_log_variance)
    elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
      # below: only log_variance is used in the KL computations
      model_variance, model_log_variance = {
        # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
        'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
        'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
      }[self.model_var_type]
      model_variance = self._extract(model_variance, t, x.shape) * tf.ones(x.shape.as_list())
      model_log_variance = self._extract(model_log_variance, t, x.shape) * tf.ones(x.shape.as_list())
    else:
      raise NotImplementedError(self.model_var_type)

    # Mean parameterization
    _maybe_clip = lambda x_: (tf.clip_by_value(x_, -1., 1.) if clip_denoised else x_)
    if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
      pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
      model_mean = model_output
    elif self.model_mean_type == 'xstart':  # the model predicts x_0
      pred_xstart = _maybe_clip(model_output)
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    elif self.model_mean_type == 'eps':  # the model predicts epsilon
      pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    else:
      raise NotImplementedError(self.model_mean_type)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    if return_pred_xstart:
      return model_mean, model_variance, model_log_variance, pred_xstart
    else:
      return model_mean, model_variance, model_log_variance

  def _predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

  def _predict_xstart_from_xprev(self, x_t, t, xprev):
    assert x_t.shape == xprev.shape
    return (  # (xprev - coef2*x_t) / coef1
        self._extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
        self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
    )

  # === Sampling ===

  def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, return_pred_xstart: bool):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      denoise_fn, x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    noise = noise_fn(shape=x.shape, dtype=x.dtype)
    assert noise.shape == x.shape
    # no noise when t == 0
    nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
    sample = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
    assert sample.shape == pred_xstart.shape
    return (sample, pred_xstart) if return_pred_xstart else sample

  def p_sample_loop(self, denoise_fn, *, shape, noise_fn=tf.random_normal):
    """
    Generate samples
    """
    assert isinstance(shape, (tuple, list))
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    img_0 = noise_fn(shape=shape, dtype=tf.float32)
    _, img_final = tf.while_loop(
      cond=lambda i_, _: tf.greater_equal(i_, 0),
      body=lambda i_, img_: [
        i_ - 1,
        self.p_sample(
          denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn, return_pred_xstart=False)
      ],
      loop_vars=[i_0, img_0],
      shape_invariants=[i_0.shape, img_0.shape],
      back_prop=False
    )
    assert img_final.shape == shape
    return img_final

  def p_sample_loop_progressive(self, denoise_fn, *, shape, noise_fn=tf.random_normal, include_xstartpred_freq=50):
    """
    Generate samples and keep track of prediction of x0
    """
    assert isinstance(shape, (tuple, list))
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    img_0 = noise_fn(shape=shape, dtype=tf.float32)  # [B, H, W, C]

    num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
    xstartpreds_0 = tf.zeros([shape[0], num_recorded_xstartpred, *shape[1:]], dtype=tf.float32)  # [B, N, H, W, C]

    def _loop_body(i_, img_, xstartpreds_):
      # Sample p(x_{t-1} | x_t) as usual
      sample, pred_xstart = self.p_sample(
        denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn, return_pred_xstart=True)
      assert sample.shape == pred_xstart.shape == shape
      # Keep track of prediction of x0
      insert_mask = tf.equal(tf.floordiv(i_, include_xstartpred_freq),
                             tf.range(num_recorded_xstartpred, dtype=tf.int32))
      insert_mask = tf.reshape(tf.cast(insert_mask, dtype=tf.float32),
                               [1, num_recorded_xstartpred, *([1] * len(shape[1:]))])  # [1, N, 1, 1, 1]
      new_xstartpreds = insert_mask * pred_xstart[:, None, ...] + (1. - insert_mask) * xstartpreds_
      return [i_ - 1, sample, new_xstartpreds]

    _, img_final, xstartpreds_final = tf.while_loop(
      cond=lambda i_, img_, xstartpreds_: tf.greater_equal(i_, 0),
      body=_loop_body,
      loop_vars=[i_0, img_0, xstartpreds_0],
      shape_invariants=[i_0.shape, img_0.shape, xstartpreds_0.shape],
      back_prop=False
    )
    assert img_final.shape == shape and xstartpreds_final.shape == xstartpreds_0.shape
    return img_final, xstartpreds_final  # xstart predictions should agree with img_final at step 0

  # === Log likelihood calculation ===

  def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, *, clip_denoised: bool, return_pred_xstart: bool):
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      denoise_fn, x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
    kl = nn.meanflat(kl) / np.log(2.)

    decoder_nll = -utils.discretized_gaussian_log_likelihood(
      x_start, means=model_mean, log_scales=0.5 * model_log_variance)
    assert decoder_nll.shape == x_start.shape
    decoder_nll = nn.meanflat(decoder_nll) / np.log(2.)

    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    assert kl.shape == decoder_nll.shape == t.shape == [x_start.shape[0]]
    output = tf.where(tf.equal(t, 0), decoder_nll, kl)
    return (output, pred_xstart) if return_pred_xstart else output

  def training_losses(self, denoise_fn, x_start, t, noise=None):
    """
    Training loss calculation
    """

    # Add noise to data
    assert t.shape == [x_start.shape[0]]
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)
    assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Calculate the loss
    if self.loss_type == 'kl':  # the variational bound
      losses = self._vb_terms_bpd(
        denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
    elif self.loss_type == 'mse':  # unweighted MSE
      assert self.model_var_type != 'learned'
      target = {
        'xprev': self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
        'xstart': x_start,
        'eps': noise
      }[self.model_mean_type]
      model_output = denoise_fn(x_t, t)
      assert model_output.shape == target.shape == x_start.shape
      losses = nn.meanflat(tf.squared_difference(target, model_output))
    else:
      raise NotImplementedError(self.loss_type)

    assert losses.shape == t.shape
    return losses

  def _prior_bpd(self, x_start):
    B, T = x_start.shape[0], self.num_timesteps
    qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=tf.fill([B], tf.constant(T - 1, dtype=tf.int32)))
    kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
    assert kl_prior.shape == x_start.shape
    return nn.meanflat(kl_prior) / np.log(2.)

  def calc_bpd_loop(self, denoise_fn, x_start, *, clip_denoised=True):
    (B, H, W, C), T = x_start.shape, self.num_timesteps

    def _loop_body(t_, cur_vals_bt_, cur_mse_bt_):
      assert t_.shape == []
      t_b = tf.fill([B], t_)
      # Calculate VLB term at the current timestep
      new_vals_b, pred_xstart = self._vb_terms_bpd(
        denoise_fn, x_start=x_start, x_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
        clip_denoised=clip_denoised, return_pred_xstart=True)
      # MSE for progressive prediction loss
      assert pred_xstart.shape == x_start.shape
      new_mse_b = nn.meanflat(tf.squared_difference(pred_xstart, x_start))
      assert new_vals_b.shape == new_mse_b.shape == [B]
      # Insert the calculated term into the tensor of all terms
      mask_bt = tf.cast(tf.equal(t_b[:, None], tf.range(T)[None, :]), dtype=tf.float32)
      new_vals_bt = cur_vals_bt_ * (1. - mask_bt) + new_vals_b[:, None] * mask_bt
      new_mse_bt = cur_mse_bt_ * (1. - mask_bt) + new_mse_b[:, None] * mask_bt
      assert mask_bt.shape == cur_vals_bt_.shape == new_vals_bt.shape == [B, T]
      return t_ - 1, new_vals_bt, new_mse_bt

    t_0 = tf.constant(T - 1, dtype=tf.int32)
    terms_0 = tf.zeros([B, T])
    mse_0 = tf.zeros([B, T])
    _, terms_bpd_bt, mse_bt = tf.while_loop(  # Note that this can be implemented with tf.map_fn instead
      cond=lambda t_, cur_vals_bt_, cur_mse_bt_: tf.greater_equal(t_, 0),
      body=_loop_body,
      loop_vars=[t_0, terms_0, mse_0],
      shape_invariants=[t_0.shape, terms_0.shape, mse_0.shape],
      back_prop=False
    )
    prior_bpd_b = self._prior_bpd(x_start)
    total_bpd_b = tf.reduce_sum(terms_bpd_bt, axis=1) + prior_bpd_b
    assert terms_bpd_bt.shape == mse_bt.shape == [B, T] and total_bpd_b.shape == prior_bpd_b.shape == [B]
    return total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt
````

## File: diffusion_tf/diffusion_utils.py
````python
import numpy as np
import tensorflow.compat.v1 as tf

from . import nn


def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  KL divergence between normal distributions parameterized by mean and log-variance.
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
                + tf.squared_difference(mean1, mean2) * tf.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


def noise_like(shape, noise_fn=tf.random_normal, repeat=False, dtype=tf.float32):
  repeat_noise = lambda: tf.repeat(noise_fn(shape=(1, *shape[1:]), dtype=dtype), repeats=shape[0], axis=0)
  noise = lambda: noise_fn(shape=shape, dtype=dtype)
  return repeat_noise() if repeat else noise()


class GaussianDiffusion:
  """
  Contains utilities for the diffusion model.
  """

  def __init__(self, *, betas, loss_type, tf_dtype=tf.float32):
    self.loss_type = loss_type

    assert isinstance(betas, np.ndarray)
    self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
    assert (betas > 0).all() and (betas <= 1).all()
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    assert alphas_cumprod_prev.shape == (timesteps,)

    self.betas = tf.constant(betas, dtype=tf_dtype)
    self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf_dtype)
    self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf_dtype)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf_dtype)
    self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1. - alphas_cumprod), dtype=tf_dtype)
    self.log_one_minus_alphas_cumprod = tf.constant(np.log(1. - alphas_cumprod), dtype=tf_dtype)
    self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod), dtype=tf_dtype)
    self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod - 1), dtype=tf_dtype)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    self.posterior_variance = tf.constant(posterior_variance, dtype=tf_dtype)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf_dtype)
    self.posterior_mean_coef1 = tf.constant(
      betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=tf_dtype)
    self.posterior_mean_coef2 = tf.constant(
      (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=tf_dtype)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    out = tf.gather(a, t)
    assert out.shape == [bs]
    return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape)
    assert noise.shape == x_start.shape
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def predict_start_from_noise(self, x_t, t, noise):
    assert x_t.shape == noise.shape
    return (
        self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

  def q_posterior(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    posterior_mean = (
        self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
            x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_losses(self, denoise_fn, x_start, t, noise=None):
    """
    Training loss calculation
    """
    B, H, W, C = x_start.shape.as_list()
    assert t.shape == [B]

    if noise is None:
      noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)
    assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    x_recon = denoise_fn(x_noisy, t)
    assert x_noisy.shape == x_start.shape
    assert x_recon.shape[:3] == [B, H, W] and len(x_recon.shape) == 4

    if self.loss_type == 'noisepred':
      # predict the noise instead of x_start. seems to be weighted naturally like SNR
      assert x_recon.shape == x_start.shape
      losses = nn.meanflat(tf.squared_difference(noise, x_recon))
    else:
      raise NotImplementedError(self.loss_type)

    assert losses.shape == [B]
    return losses

  def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool):
    if self.loss_type == 'noisepred':
      x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
    else:
      raise NotImplementedError(self.loss_type)

    if clip_denoised:
      x_recon = tf.clip_by_value(x_recon, -1., 1.)

    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    assert model_mean.shape == x_recon.shape == x.shape
    assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
    return model_mean, posterior_variance, posterior_log_variance

  def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
    noise = noise_like(x.shape, noise_fn, repeat_noise)
    assert noise.shape == x.shape
    # no noise when t == 0
    nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
    return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

  def p_sample_loop(self, denoise_fn, *, shape, noise_fn=tf.random_normal):
    """
    Generate samples
    """
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    assert isinstance(shape, (tuple, list))
    img_0 = noise_fn(shape=shape, dtype=tf.float32)
    _, img_final = tf.while_loop(
      cond=lambda i_, _: tf.greater_equal(i_, 0),
      body=lambda i_, img_: [
        i_ - 1,
        self.p_sample(denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn)
      ],
      loop_vars=[i_0, img_0],
      shape_invariants=[i_0.shape, img_0.shape],
      back_prop=False
    )
    assert img_final.shape == shape
    return img_final

  def p_sample_loop_trajectory(self, denoise_fn, *, shape, noise_fn=tf.random_normal, repeat_noise_steps=-1):
    """
    Generate samples, returning intermediate images
    Useful for visualizing how denoised images evolve over time
    Args:
      repeat_noise_steps (int): Number of denoising timesteps in which the same noise
        is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
    """
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    assert isinstance(shape, (tuple, list))
    img_0 = noise_like(shape, noise_fn, repeat_noise_steps >= 0)
    times = tf.Variable([i_0])
    imgs = tf.Variable([img_0])
    # Steps with repeated noise
    times, imgs = tf.while_loop(
      cond=lambda times_, _: tf.less_equal(self.num_timesteps - times_[-1], repeat_noise_steps),
      body=lambda times_, imgs_: [
        tf.concat([times_, [times_[-1] - 1]], 0),
        tf.concat([imgs_, [self.p_sample(denoise_fn=denoise_fn,
                                         x=imgs_[-1],
                                         t=tf.fill([shape[0]], times_[-1]),
                                         noise_fn=noise_fn,
                                         repeat_noise=True)]], 0)
      ],
      loop_vars=[times, imgs],
      shape_invariants=[tf.TensorShape([None, *i_0.shape]),
                        tf.TensorShape([None, *img_0.shape])],
      back_prop=False
    )
    # Steps with different noise for each batch element
    times, imgs = tf.while_loop(
      cond=lambda times_, _: tf.greater_equal(times_[-1], 0),
      body=lambda times_, imgs_: [
        tf.concat([times_, [times_[-1] - 1]], 0),
        tf.concat([imgs_, [self.p_sample(denoise_fn=denoise_fn,
                                         x=imgs_[-1],
                                         t=tf.fill([shape[0]], times_[-1]),
                                         noise_fn=noise_fn,
                                         repeat_noise=False)]], 0)
      ],
      loop_vars=[times, imgs],
      shape_invariants=[tf.TensorShape([None, *i_0.shape]),
                        tf.TensorShape([None, *img_0.shape])],
      back_prop=False
    )
    assert imgs[-1].shape == shape
    return times, imgs

  def interpolate(self, denoise_fn, *, shape, noise_fn=tf.random_normal):
    """
    Interpolate between images.
    t == 0 means diffuse images for 1 timestep before mixing.
    """
    assert isinstance(shape, (tuple, list))

    # Placeholders for real samples to interpolate
    x1 = tf.placeholder(tf.float32, shape)
    x2 = tf.placeholder(tf.float32, shape)
    # lam == 0.5 averages diffused images.
    lam = tf.placeholder(tf.float32, shape=())
    t = tf.placeholder(tf.int32, shape=())

    # Add noise via forward diffusion
    # TODO: use the same noise for both endpoints?
    # t_batched = tf.constant([t] * x1.shape[0], dtype=tf.int32)
    t_batched = tf.stack([t] * x1.shape[0])
    xt1 = self.q_sample(x1, t=t_batched)
    xt2 = self.q_sample(x2, t=t_batched)

    # Mix latents
    # Linear interpolation
    xt_interp = (1 - lam) * xt1 + lam * xt2
    # Constant variance interpolation
    # xt_interp = tf.sqrt(1 - lam * lam) * xt1 + lam * xt2

    # Reverse diffusion (similar to self.p_sample_loop)
    # t = tf.constant(t, dtype=tf.int32)
    _, x_interp = tf.while_loop(
      cond=lambda i_, _: tf.greater_equal(i_, 0),
      body=lambda i_, img_: [
        i_ - 1,
        self.p_sample(denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn)
      ],
      loop_vars=[t, xt_interp],
      shape_invariants=[t.shape, xt_interp.shape],
      back_prop=False
    )
    assert x_interp.shape == shape

    return x1, x2, lam, x_interp, t
````

## File: diffusion_tf/nn.py
````python
import math
import string

import tensorflow.compat.v1 as tf

# ===== Neural network building defaults =====
DEFAULT_DTYPE = tf.float32


def default_init(scale):
  return tf.initializers.variance_scaling(scale=1e-10 if scale == 0 else scale, mode='fan_avg', distribution='uniform')


# ===== Utilities =====

def _wrapped_print(x, *args, **kwargs):
  print_op = tf.print(*args, **kwargs)
  with tf.control_dependencies([print_op]):
    return tf.identity(x)


def debug_print(x, name):
  return _wrapped_print(x, name, tf.reduce_mean(x), tf.math.reduce_std(x), tf.reduce_min(x), tf.reduce_max(x))


def flatten(x):
  return tf.reshape(x, [int(x.shape[0]), -1])


def sumflat(x):
  return tf.reduce_sum(x, axis=list(range(1, len(x.shape))))


def meanflat(x):
  return tf.reduce_mean(x, axis=list(range(1, len(x.shape))))


# ===== Neural network layers =====

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


def nin(x, *, name, num_units, init_scale=1.):
  with tf.variable_scope(name):
    in_dim = int(x.shape[-1])
    W = tf.get_variable('W', shape=[in_dim, num_units], initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + [num_units]
    return y


def dense(x, *, name, num_units, init_scale=1., bias=True):
  with tf.variable_scope(name):
    _, in_dim = x.shape
    W = tf.get_variable('W', shape=[in_dim, num_units], initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    z = tf.matmul(x, W)
    if not bias:
      return z
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    return z + b


def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=1, dilation=None, pad='SAME', init_scale=1., bias=True):
  with tf.variable_scope(name):
    assert x.shape.ndims == 4
    if isinstance(filter_size, int):
      filter_size = (filter_size, filter_size)
    W = tf.get_variable('W', shape=[*filter_size, int(x.shape[-1]), num_units],
                        initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    z = tf.nn.conv2d(x, W, strides=stride, padding=pad, dilations=dilation)
    if not bias:
      return z
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    return z + b


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = tf.exp(tf.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
  # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
    emb = tf.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == [timesteps.shape[0], embedding_dim]
  return emb
````

## File: diffusion_tf/utils.py
````python
import contextlib
import io
import random
import time

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from tensorflow.compat.v1 import gfile
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import Event


class SummaryWriter:
  """Tensorflow summary writer inspired by Jaxboard.
  This version doesn't try to avoid Tensorflow dependencies, because this
  project uses Tensorflow.
  """

  def __init__(self, dir, write_graph=True):
    if not gfile.IsDirectory(dir):
      gfile.MakeDirs(dir)
    self.writer = tf.summary.FileWriter(
      dir, graph=tf.get_default_graph() if write_graph else None)

  def flush(self):
    self.writer.flush()

  def close(self):
    self.writer.close()

  def _write_event(self, summary_value, step):
    self.writer.add_event(
      Event(
        wall_time=round(time.time()),
        step=step,
        summary=Summary(value=[summary_value])))

  def scalar(self, tag, value, step):
    self._write_event(Summary.Value(tag=tag, simple_value=float(value)), step)

  def image(self, tag, image, step):
    image = np.asarray(image)
    if image.ndim == 2:
      image = image[:, :, None]
    if image.shape[-1] == 1:
      image = np.repeat(image, 3, axis=-1)

    bytesio = io.BytesIO()
    Image.fromarray(image).save(bytesio, 'PNG')
    image_summary = Summary.Image(
      encoded_image_string=bytesio.getvalue(),
      colorspace=3,
      height=image.shape[0],
      width=image.shape[1])
    self._write_event(Summary.Value(tag=tag, image=image_summary), step)

  def images(self, tag, images, step):
    self.image(tag, tile_imgs(images), step=step)


def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)


def tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
  assert pad_pixels >= 0 and 0 <= pad_val <= 255

  imgs = np.asarray(imgs)
  assert imgs.dtype == np.uint8
  if imgs.ndim == 3:
    imgs = imgs[..., None]
  n, h, w, c = imgs.shape
  assert c == 1 or c == 3, 'Expected 1 or 3 channels'

  if num_col <= 0:
    # Make a square
    ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
    num_row = ceil_sqrt_n
    num_col = ceil_sqrt_n
  else:
    # Make a B/num_per_row x num_per_row grid
    assert n % num_col == 0
    num_row = int(np.ceil(n / num_col))

  imgs = np.pad(
    imgs,
    pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
    mode='constant',
    constant_values=pad_val
  )
  h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
  imgs = imgs.reshape(num_row, num_col, h, w, c)
  imgs = imgs.transpose(0, 2, 1, 3, 4)
  imgs = imgs.reshape(num_row * h, num_col * w, c)

  if pad_pixels > 0:
    imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
  if c == 1:
    imgs = imgs[..., 0]
  return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
  Image.fromarray(tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)).save(filename)


# ===

def approx_standard_normal_cdf(x):
  return 0.5 * (1.0 + tf.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
  # Assumes data is integers [0, 255] rescaled to [-1, 1]
  assert x.shape == means.shape == log_scales.shape
  centered_x = x - means
  inv_stdv = tf.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + 1. / 255.)
  cdf_plus = approx_standard_normal_cdf(plus_in)
  min_in = inv_stdv * (centered_x - 1. / 255.)
  cdf_min = approx_standard_normal_cdf(min_in)
  log_cdf_plus = tf.log(tf.maximum(cdf_plus, 1e-12))
  log_one_minus_cdf_min = tf.log(tf.maximum(1. - cdf_min, 1e-12))
  cdf_delta = cdf_plus - cdf_min
  log_probs = tf.where(
    x < -0.999, log_cdf_plus,
    tf.where(x > 0.999, log_one_minus_cdf_min,
             tf.log(tf.maximum(cdf_delta, 1e-12))))
  assert log_probs.shape == x.shape
  return log_probs


# ===


def rms(variables):
  return tf.sqrt(
    sum([tf.reduce_sum(tf.square(v)) for v in variables]) /
    sum(int(np.prod(v.shape.as_list())) for v in variables))


def get_warmed_up_lr(max_lr, warmup, global_step):
  if warmup == 0:
    return max_lr
  return max_lr * tf.minimum(tf.cast(global_step, tf.float32) / float(warmup), 1.0)


def make_optimizer(
    *,
    loss, trainable_variables, global_step, tpu: bool,
    optimizer: str, lr: float, grad_clip: float,
    rmsprop_decay=0.95, rmsprop_momentum=0.9, epsilon=1e-8
):
  if optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
      learning_rate=lr, epsilon=epsilon)
  elif optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=lr, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=epsilon)
  else:
    raise NotImplementedError(optimizer)

  if tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  # compute gradient
  grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_variables)

  # clip gradient
  clipped_grads, gnorm = tf.clip_by_global_norm([g for (g, _) in grads_and_vars], grad_clip)
  grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

  # train
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
  return train_op, gnorm


@contextlib.contextmanager
def ema_scope(orig_model_ema):
  def _ema_getter(getter, name, *args, **kwargs):
    v = getter(name, *args, **kwargs)
    v = orig_model_ema.average(v)
    if v is None:
      raise RuntimeError('Variable {} has no EMA counterpart'.format(name))
    return v

  with tf.variable_scope(tf.get_variable_scope(), custom_getter=_ema_getter, reuse=True):
    with tf.name_scope('ema_scope'):
      yield


def get_gcp_region():
  # https://stackoverflow.com/a/31689692
  import requests
  metadata_server = "http://metadata/computeMetadata/v1/instance/"
  metadata_flavor = {'Metadata-Flavor': 'Google'}
  zone = requests.get(metadata_server + 'zone', headers=metadata_flavor).text
  zone = zone.split('/')[-1]
  region = '-'.join(zone.split('-')[:-1])
  return region
````

## File: scripts/run_celebahq.py
````python
"""
CelebaHQ 256x256

python3 scripts/run_celebahq.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME
python3 scripts/run_celebahq.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR
"""

import functools

import fire
import numpy as np
import tensorflow.compat.v1 as tf

from diffusion_tf import utils
from diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion
from diffusion_tf.models import unet
from diffusion_tf.tpu_utils import tpu_utils, datasets


class Model(tpu_utils.Model):
  def __init__(self, *, model_name, betas: np.ndarray, loss_type: str, num_classes: int,
               dropout: float, randflip, block_size: int):
    self.model_name = model_name
    self.diffusion = GaussianDiffusion(betas=betas, loss_type=loss_type)
    self.num_classes = num_classes
    self.dropout = dropout
    self.randflip = randflip
    self.block_size = block_size

  def _denoise(self, x, t, y, dropout):
    B, H, W, C = x.shape.as_list()
    assert x.dtype == tf.float32
    assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
    assert y.shape == [B] and y.dtype in [tf.int32, tf.int64]
    orig_out_ch = out_ch = C

    if self.block_size != 1:  # this can be used to reduce memory consumption
      x = tf.nn.space_to_depth(x, self.block_size)
      out_ch *= self.block_size ** 2

    y = None
    if self.model_name == 'unet2d16b2c112244':  # 114M for block_size=1
      out = unet.model(
        x, t=t, y=y, name='model', ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_resolutions=(16,),
        out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
      )
    else:
      raise NotImplementedError(self.model_name)

    if self.block_size != 1:
      out = tf.nn.depth_to_space(out, self.block_size)
    assert out.shape == [B, H, W, orig_out_ch]
    return out

  def train_fn(self, x, y):
    B, H, W, C = x.shape
    if self.randflip:
      x = tf.image.random_flip_left_right(x)
      assert x.shape == [B, H, W, C]
    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
    losses = self.diffusion.p_losses(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)
    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

  def samples_fn(self, dummy_noise, y):
    return {
      'samples': self.diffusion.p_sample_loop(
        denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
        shape=dummy_noise.shape.as_list(),
        noise_fn=tf.random_normal
      )
    }

  def samples_fn_denoising_trajectory(self, dummy_noise, y, repeat_noise_steps=0):
    times, imgs = self.diffusion.p_sample_loop_trajectory(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
      shape=dummy_noise.shape.as_list(),
      noise_fn=tf.random_normal,
      repeat_noise_steps=repeat_noise_steps
    )
    return {
      'samples': imgs[-1],
      'denoising_trajectory_times': times,
      'denoising_trajectory_images': imgs
    }

  def interpolate_fn(self, dummy_noise, y):
    x1, x2, lam, x_interp, t = self.diffusion.interpolate(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
      shape=dummy_noise.shape.as_list(),
      noise_fn=tf.random_normal,
    )
    return {
      'x1': x1,    # placeholder
      'x2': x2,    # placeholder
      'lam': lam,  # placeholder
      't': t,      # placeholder
      'x_interp': x_interp
    }


def evaluation(
    model_dir, tpu_name, bucket_name_prefix, once=False, dump_samples_only=False, total_bs=128,
    tfds_data_dir='tensorflow_datasets',
):
  region = utils.get_gcp_region()
  tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
  kwargs = tpu_utils.load_train_kwargs(model_dir)
  print('loaded kwargs:', kwargs)
  ds = datasets.get_dataset(kwargs['dataset'], tfds_data_dir=tfds_data_dir)
  worker = tpu_utils.EvalWorker(
    tpu_name=tpu_name,
    model_constructor=lambda: Model(
      model_name=kwargs['model_name'],
      betas=get_beta_schedule(
        kwargs['beta_schedule'], beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'],
        num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
      ),
      loss_type=kwargs['loss_type'],
      num_classes=ds.num_classes,
      dropout=kwargs['dropout'],
      randflip=kwargs['randflip'],
      block_size=kwargs['block_size']
    ),
    total_bs=total_bs, inception_bs=total_bs, num_inception_samples=2048,
    dataset=ds,
  )
  worker.run(logdir=model_dir, once=once, skip_non_ema_pass=True, dump_samples_only=dump_samples_only)


def train(
    exp_name, tpu_name, bucket_name_prefix, model_name='unet2d16b2c112244', dataset='celebahq256',
    optimizer='adam', total_bs=64, grad_clip=1., lr=0.00002, warmup=5000,
    num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear', loss_type='noisepred',
    dropout=0.0, randflip=1, block_size=1,
    tfds_data_dir='tensorflow_datasets', log_dir='logs'
):
  region = utils.get_gcp_region()
  tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
  log_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, log_dir)
  kwargs = dict(locals())
  ds = datasets.get_dataset(dataset, tfds_data_dir=tfds_data_dir)
  tpu_utils.run_training(
    date_str='9999-99-99',
    exp_name='{exp_name}_{dataset}_{model_name}_{optimizer}_bs{total_bs}_lr{lr}w{warmup}_beta{beta_start}-{beta_end}-{beta_schedule}_t{num_diffusion_timesteps}_{loss_type}_dropout{dropout}_randflip{randflip}_blk{block_size}'.format(
      **kwargs),
    model_constructor=lambda: Model(
      model_name=model_name,
      betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
      ),
      loss_type=loss_type,
      num_classes=ds.num_classes,
      dropout=dropout,
      randflip=randflip,
      block_size=block_size
    ),
    optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup, grad_clip=grad_clip,
    train_input_fn=ds.train_input_fn,
    tpu=tpu_name, log_dir=log_dir, dump_kwargs=kwargs
  )


if __name__ == '__main__':
  fire.Fire()
````

## File: scripts/run_cifar.py
````python
"""
Unconditional CIFAR10

python3 scripts/run_cifar.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME
python3 scripts/run_cifar.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR
"""

import functools

import fire
import numpy as np
import tensorflow.compat.v1 as tf

from diffusion_tf import utils
from diffusion_tf.diffusion_utils_2 import get_beta_schedule, GaussianDiffusion2
from diffusion_tf.models import unet
from diffusion_tf.tpu_utils import tpu_utils, datasets, simple_eval_worker


class Model(tpu_utils.Model):
  def __init__(self, *, model_name, betas: np.ndarray, model_mean_type: str, model_var_type: str, loss_type: str,
               num_classes: int, dropout: float, randflip):
    self.model_name = model_name
    self.diffusion = GaussianDiffusion2(
      betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type)
    self.num_classes = num_classes
    self.dropout = dropout
    self.randflip = randflip

  def _denoise(self, x, t, y, dropout):
    B, H, W, C = x.shape.as_list()
    assert x.dtype == tf.float32
    assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
    assert y.shape == [B] and y.dtype in [tf.int32, tf.int64]
    out_ch = (C * 2) if self.diffusion.model_var_type == 'learned' else C
    y = None
    if self.model_name == 'unet2d16b2':  # 35.7M
      return unet.model(
        x, t=t, y=y, name='model', ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
        out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
      )
    raise NotImplementedError(self.model_name)

  def train_fn(self, x, y):
    B, H, W, C = x.shape
    if self.randflip:
      x = tf.image.random_flip_left_right(x)
      assert x.shape == [B, H, W, C]
    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
    losses = self.diffusion.training_losses(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)
    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

  def samples_fn(self, dummy_noise, y):
    return {
      'samples': self.diffusion.p_sample_loop(
        denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
        shape=dummy_noise.shape.as_list(),
        noise_fn=tf.random_normal
      )
    }

  def progressive_samples_fn(self, dummy_noise, y):
    samples, progressive_samples = self.diffusion.p_sample_loop_progressive(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
      shape=dummy_noise.shape.as_list(),
      noise_fn=tf.random_normal
    )
    return {'samples': samples, 'progressive_samples': progressive_samples}

  def bpd_fn(self, x, y):
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
      x_start=x
    )
    return {
      'total_bpd': total_bpd_b,
      'terms_bpd': terms_bpd_bt,
      'prior_bpd': prior_bpd_b,
      'mse': mse_bt
    }


def _load_model(kwargs, ds):
  return Model(
    model_name=kwargs['model_name'],
    betas=get_beta_schedule(
      kwargs['beta_schedule'], beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'],
      num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
    ),
    model_mean_type=kwargs['model_mean_type'],
    model_var_type=kwargs['model_var_type'],
    loss_type=kwargs['loss_type'],
    num_classes=ds.num_classes,
    dropout=kwargs['dropout'],
    randflip=kwargs['randflip']
  )


def simple_eval(model_dir, tpu_name, bucket_name_prefix, mode, load_ckpt, total_bs=256, tfds_data_dir='tensorflow_datasets'):
  region = utils.get_gcp_region()
  tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
  kwargs = tpu_utils.load_train_kwargs(model_dir)
  print('loaded kwargs:', kwargs)
  ds = datasets.get_dataset(kwargs['dataset'], tfds_data_dir=tfds_data_dir)
  worker = simple_eval_worker.SimpleEvalWorker(
    tpu_name=tpu_name, model_constructor=functools.partial(_load_model, kwargs=kwargs, ds=ds),
    total_bs=total_bs, dataset=ds)
  worker.run(mode=mode, logdir=model_dir, load_ckpt=load_ckpt)


def evaluation(  # evaluation loop for use during training
    model_dir, tpu_name, bucket_name_prefix, once=False, dump_samples_only=False, total_bs=256,
    tfds_data_dir='tensorflow_datasets', load_ckpt=None
):
  region = utils.get_gcp_region()
  tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
  kwargs = tpu_utils.load_train_kwargs(model_dir)
  print('loaded kwargs:', kwargs)
  ds = datasets.get_dataset(kwargs['dataset'], tfds_data_dir=tfds_data_dir)
  worker = tpu_utils.EvalWorker(
    tpu_name=tpu_name,
    model_constructor=functools.partial(_load_model, kwargs=kwargs, ds=ds),
    total_bs=total_bs, inception_bs=total_bs, num_inception_samples=50000,
    dataset=ds,
  )
  worker.run(
    logdir=model_dir, once=once, skip_non_ema_pass=True, dump_samples_only=dump_samples_only, load_ckpt=load_ckpt)


def train(
    exp_name, tpu_name, bucket_name_prefix, model_name='unet2d16b2', dataset='cifar10',
    optimizer='adam', total_bs=128, grad_clip=1., lr=2e-4, warmup=5000,
    num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear',
    model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse',
    dropout=0.1, randflip=1,
    tfds_data_dir='tensorflow_datasets', log_dir='logs', keep_checkpoint_max=2
):
  region = utils.get_gcp_region()
  tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
  log_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, log_dir)
  kwargs = dict(locals())
  ds = datasets.get_dataset(dataset, tfds_data_dir=tfds_data_dir)
  tpu_utils.run_training(
    date_str='9999-99-99',
    exp_name='{exp_name}_{dataset}_{model_name}_{optimizer}_bs{total_bs}_lr{lr}w{warmup}_beta{beta_start}-{beta_end}-{beta_schedule}_t{num_diffusion_timesteps}_{model_mean_type}-{model_var_type}-{loss_type}_dropout{dropout}_randflip{randflip}'.format(
      **kwargs),
    model_constructor=lambda: Model(
      model_name=model_name,
      betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
      ),
      model_mean_type=model_mean_type,
      model_var_type=model_var_type,
      loss_type=loss_type,
      num_classes=ds.num_classes,
      dropout=dropout,
      randflip=randflip
    ),
    optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup, grad_clip=grad_clip,
    train_input_fn=ds.train_input_fn,
    tpu=tpu_name, log_dir=log_dir, dump_kwargs=kwargs, iterations_per_loop=2000, keep_checkpoint_max=keep_checkpoint_max
  )


if __name__ == '__main__':
  fire.Fire()
````

## File: scripts/run_lsun.py
````python
"""
LSUN church, bedroom and cat 256x256

# LSUN church
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --tpu_name $TPU_NAME --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME --tfr_file 'tensorflow_datasets/lsun/church/church-r08.tfrecords'
python3 scripts/run_lsun.py evaluation --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/church/church-r08.tfrecords'

# LSUN bedroom
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tfr_file 'tensorflow_datasets/lsun/bedroom-full/bedroom-full-r08.tfrecords'
python3 scripts/run_lsun.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/bedroom-full/bedroom-full-r08.tfrecords'

# LSUN cat
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME --tfr_file 'tensorflow_datasets/lsun/cat/cat-r08.tfrecords' --randflip 0
python3 scripts/run_lsun.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/cat/cat-r08.tfrecords'
"""

import functools

import fire
import numpy as np
import tensorflow.compat.v1 as tf

from diffusion_tf import utils
from diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion
from diffusion_tf.models import unet
from diffusion_tf.tpu_utils import tpu_utils, datasets


class Model(tpu_utils.Model):
  def __init__(self, *, model_name, betas: np.ndarray, loss_type: str, num_classes: int,
               dropout: float, randflip, block_size: int):
    self.model_name = model_name
    self.diffusion = GaussianDiffusion(betas=betas, loss_type=loss_type)
    self.num_classes = num_classes
    self.dropout = dropout
    self.randflip = randflip
    self.block_size = block_size

  def _denoise(self, x, t, y, dropout):
    B, H, W, C = x.shape.as_list()
    assert x.dtype == tf.float32
    assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
    assert y.shape == [B] and y.dtype in [tf.int32, tf.int64]
    orig_out_ch = out_ch = C

    if self.block_size != 1:
      x = tf.nn.space_to_depth(x, self.block_size)
      out_ch *= self.block_size ** 2

    y = None
    if self.model_name == 'unet2d16b2c112244':  # 114M for block_size=1
      out = unet.model(
        x, t=t, y=y, name='model', ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_resolutions=(16,),
        out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
      )
    else:
      raise NotImplementedError(self.model_name)

    if self.block_size != 1:
      out = tf.nn.depth_to_space(out, self.block_size)
    assert out.shape == [B, H, W, orig_out_ch]
    return out

  def train_fn(self, x, y):
    B, H, W, C = x.shape
    if self.randflip:
      x = tf.image.random_flip_left_right(x)
      assert x.shape == [B, H, W, C]
    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
    losses = self.diffusion.p_losses(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)
    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

  def samples_fn(self, dummy_noise, y):
    return {
      'samples': self.diffusion.p_sample_loop(
        denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
        shape=dummy_noise.shape.as_list(),
        noise_fn=tf.random_normal
      )
    }


def evaluation(
    model_dir, tpu_name, bucket_name_prefix, once=False, dump_samples_only=False, total_bs=128,
    tfr_file='tensorflow_datasets/lsun/church-r08.tfrecords', samples_dir=None, num_inception_samples=2048,
):
  region = utils.get_gcp_region()
  tfr_file = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfr_file)
  kwargs = tpu_utils.load_train_kwargs(model_dir)
  print('loaded kwargs:', kwargs)
  ds = datasets.get_dataset(kwargs['dataset'], tfr_file=tfr_file)
  worker = tpu_utils.EvalWorker(
    tpu_name=tpu_name,
    model_constructor=lambda: Model(
      model_name=kwargs['model_name'],
      betas=get_beta_schedule(
        kwargs['beta_schedule'], beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'],
        num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
      ),
      loss_type=kwargs['loss_type'],
      num_classes=ds.num_classes,
      dropout=kwargs['dropout'],
      randflip=kwargs['randflip'],
      block_size=kwargs['block_size']
    ),
    total_bs=total_bs, inception_bs=total_bs, num_inception_samples=num_inception_samples,
    dataset=ds,
    limit_dataset_size=30000  # limit size of dataset for computing Inception features, for memory reasons
  )
  worker.run(logdir=model_dir, once=once, skip_non_ema_pass=True, dump_samples_only=dump_samples_only,
             samples_dir=samples_dir)


def train(
    exp_name, tpu_name, bucket_name_prefix, model_name='unet2d16b2c112244', dataset='lsun',
    optimizer='adam', total_bs=64, grad_clip=1., lr=2e-5, warmup=5000,
    num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear', loss_type='noisepred',
    dropout=0.0, randflip=1, block_size=1,
    tfr_file='tensorflow_datasets/lsun/church/church-r08.tfrecords', log_dir='logs',
    warm_start_model_dir=None
):
  region = utils.get_gcp_region()
  tfr_file = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfr_file)
  log_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, log_dir)
  print("tfr_file:", tfr_file)
  print("log_dir:", log_dir)
  kwargs = dict(locals())
  ds = datasets.get_dataset(dataset, tfr_file=tfr_file)
  tpu_utils.run_training(
    date_str='9999-99-99',
    exp_name='{exp_name}_{dataset}_{model_name}_{optimizer}_bs{total_bs}_lr{lr}w{warmup}_beta{beta_start}-{beta_end}-{beta_schedule}_t{num_diffusion_timesteps}_{loss_type}_dropout{dropout}_randflip{randflip}_blk{block_size}'.format(
      **kwargs),
    model_constructor=lambda: Model(
      model_name=model_name,
      betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
      ),
      loss_type=loss_type,
      num_classes=ds.num_classes,
      dropout=dropout,
      randflip=randflip,
      block_size=block_size
    ),
    optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup, grad_clip=grad_clip,
    train_input_fn=ds.train_input_fn,
    tpu=tpu_name, log_dir=log_dir, dump_kwargs=kwargs,
    warm_start_from=tf.estimator.WarmStartSettings(
      ckpt_to_initialize_from=tf.train.latest_checkpoint(warm_start_model_dir),
      vars_to_warm_start=[".*"]
    ) if warm_start_model_dir else None
  )


if __name__ == '__main__':
  fire.Fire()
````

## File: .gitignore
````
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

.idea
````

## File: README.md
````markdown
# Denoising Diffusion Probabilistic Models

Jonathan Ho, Ajay Jain, Pieter Abbeel

Paper: https://arxiv.org/abs/2006.11239

Website: https://hojonathanho.github.io/diffusion

![Samples generated by our model](resources/samples.png)

Experiments run on Google Cloud TPU v3-8.
Requires TensorFlow 1.15 and Python 3.5, and these dependencies for CPU instances (see `requirements.txt`):
```
pip3 install fire
pip3 install scipy
pip3 install pillow
pip3 install tensorflow-probability==0.8
pip3 install tensorflow-gan==0.0.0.dev0
pip3 install tensorflow-datasets==2.1.0
```

The training and evaluation scripts are in the `scripts/` subdirectory.
The commands to run training and evaluation are in comments at the top of the scripts.
Data is stored in GCS buckets. The scripts are written to assume that the bucket names are of the form `gs://mybucketprefix-us-central1`; i.e. some prefix followed by the region.
The prefix should be passed into the scripts using the `--bucket_name_prefix` flag.

Models and samples can be found at: https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja

## Citation
If you find our work relevant to your research, please cite:
```
@article{ho2020denoising,
    title={Denoising Diffusion Probabilistic Models},
    author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year={2020},
    journal={arXiv preprint arxiv:2006.11239}
}
```
````

## File: requirements.txt
````
absl-py==0.9.0
appdirs==1.4.3
astor==0.8.1
attrs==19.3.0
backcall==0.1.0
bleach==3.1.0
cachetools==4.0.0
certifi==2019.11.28
chardet==3.0.4
cloud-tpu-client==0.5
cloud-tpu-profiler==1.15.0rc1
cloudpickle==1.1.1
cycler==0.10.0
decorator==4.4.1
defusedxml==0.6.0
dill==0.3.1.1
distlib==0.3.0
distro==1.0.1
entrypoints==0.3
filelock==3.0.12
fire==0.2.1
future==0.18.2
gast==0.2.2
google-api-python-client==1.7.11
google-auth==1.11.2
google-auth-httplib2==0.0.3
google-compute-engine==20191210.0
google-pasta==0.1.8
googleapis-common-protos==1.51.0
grpcio==1.27.2
h5py==2.10.0
httplib2==0.17.0
idna==2.8
importlib-metadata==1.5.0
importlib-resources==1.0.2
ipykernel==5.1.4
ipython==7.9.0
ipython-genutils==0.2.0
ipywidgets==7.5.1
jedi==0.16.0
Jinja2==2.11.1
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==5.3.4
jupyter-console==6.1.0
jupyter-core==4.6.3
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.1.0
Markdown==3.2.1
MarkupSafe==1.1.1
matplotlib==3.0.3
mistune==0.8.4
nbconvert==5.6.1
nbformat==5.0.4
notebook==6.0.3
numpy==1.18.1
oauth2client==4.1.3
opt-einsum==3.1.0
pandas==0.25.3
pandocfilters==1.4.2
parso==0.6.1
pexpect==4.8.0
pickleshare==0.7.5
Pillow==7.0.0
prometheus-client==0.7.1
promise==2.3
prompt-toolkit==2.0.10
protobuf==3.11.3
psutil==5.7.0
ptyprocess==0.6.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycurl==7.43.0
Pygments==2.5.2
pygobject==3.22.0
pyparsing==2.4.6
pyrsistent==0.15.7
python-apt==1.4.1
python-dateutil==2.8.1
pytz==2019.3
PyYAML==5.3
pyzmq==18.1.1
qtconsole==4.6.0
requests==2.22.0
rsa==4.0
scipy==1.4.1
seaborn==0.9.1
Send2Trash==1.5.0
six==1.14.0
tensorboard==1.15.0
tensorflow==1.15.0
tensorflow-datasets==2.1.0
tensorflow-estimator==1.15.1
tensorflow-gan==0.0.0.dev0
tensorflow-hub==0.7.0
tensorflow-metadata==0.21.1
tensorflow-probability==0.8.0
tensorflow-serving-api==1.14.0
termcolor==1.1.0
terminado==0.8.3
testpath==0.4.4
tornado==6.0.3
tqdm==4.42.1
traitlets==4.3.3
unattended-upgrades==0.1
uritemplate==3.0.1
urllib3==1.25.7
virtualenv==20.0.4
wcwidth==0.1.8
webencodings==0.5.1
Werkzeug==1.0.0
widgetsnbextension==3.5.1
wrapt==1.12.0
zipp==1.2.0
````
