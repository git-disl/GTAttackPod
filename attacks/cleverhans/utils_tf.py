from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import os
import six
import tensorflow as tf
import time
import warnings

from .utils import batch_indices, _ArgsWrapper

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class _FlagsWrapper(_ArgsWrapper):
    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """

    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def tf_model_train(*args, **kwargs):
    warnings.warn("`tf_model_train` is deprecated. Switch to `model_train`."
                  "`tf_model_train` will be removed after 2017-07-18.")
    return model_train(*args, **kwargs)


def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, init_all=True, evaluate=None,
                verbose=True, args=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controlling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param init_all: (boolean) If set to true, all TF variables in the session
                     are (re)initialized, otherwise only previously
                     uninitialized variables are initialized before training.
    :param evaluate: function that is run after each training iteration
                     (typically to display the test/validation accuracy).
    :param verbose: (boolean) all print statements disabled when set to False.
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'train_dir'
                 and 'filename'
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + model_loss(y, predictions_adv)) / 2

    train_step = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate,
                                            rho=0.95,
                                            epsilon=1e-08).minimize(loss)

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            if init_all:
                tf.global_variables_initializer().run()
            else:
                initialize_uninitialized_global_variables(sess)
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "cleverhans may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        for epoch in six.moves.xrange(args.nb_epochs):
            if verbose:
                print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            prev = time.time()
            for batch in range(nb_batches):
                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end]})
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                print("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate()

        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and saved at:" + str(save_path))
        else:
            print("Completed model training.")

    return True


def tf_model_eval(*args, **kwargs):
    warnings.warn("`tf_model_eval` is deprecated. Switch to `model_eval`."
                  "`tf_model_eval` will be removed after 2017-07-18.")
    return model_eval(*args, **kwargs)


def model_eval(sess, x, y, model, X_test, Y_test, args=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    # Define accuracy symbolically
    correct_preds = tf.equal(tf.argmax(y, axis=-1), tf.argmax(model, axis=-1))
    acc_value = tf.reduce_mean(tf.to_float(correct_preds))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            cur_acc = acc_value.eval(
                feed_dict={x: X_test[start:end],
                           y: Y_test[start:end]})

            accuracy += (cur_batch_size * cur_acc)

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy


def tf_model_load(sess):
    """

    :param sess:
    :param x:
    :param y:
    :param model:
    :return:
    """
    with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))

    return True


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, args=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.

    :param sess:
    :param tf_inputs:
    :param tf_outputs:
    :param numpy_inputs:
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in six.moves.xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in six.moves.xrange(0, m, args.batch_size):
            batch = start // args.batch_size
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * args.batch_size
            end = start + args.batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= args.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def model_argmax(sess, x, predictions, samples):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def assert_less_equal(*args, **kwargs):
    """
    Wrapper for tf.assert_less_equal
    Overrides tf.device so that the assert always goes on CPU.
    The unwrapped version raises an exception if used with tf.device("/GPU:x").
    """
    with tf.device("/CPU:0"):
        return tf.assert_less_equal(*args, **kwargs)


def assert_greater_equal(*args, **kwargs):
    """
    Wrapper for tf.assert_greater_equal.
    Overrides tf.device so that the assert always goes on CPU.
    The unwrapped version raises an exception if used with tf.device("/GPU:x").
    """
    with tf.device("/CPU:0"):
        return tf.assert_greater_equal(*args, **kwargs)


def op_with_scalar_cast(a, b, f):
    """
    Builds the graph to compute f(a, b).
    If only one of the two arguments is a scalar and the operation would
    cause a type error without casting, casts the scalar to match the
    tensor.
    :param a: a tf-compatible array or scalar
    :param b: a tf-compatible array or scalar
    """

    try:
        return f(a, b)
    except (TypeError, ValueError):
        pass

    def is_scalar(x):
        """Return True if `x` is a scalar"""
        if hasattr(x, "get_shape"):
            shape = x.get_shape()
            return shape.ndims == 0
        if hasattr(x, "ndim"):
            return x.ndim == 0
        assert isinstance(x, (int, float))
        return True

    a_scalar = is_scalar(a)
    b_scalar = is_scalar(b)

    if a_scalar and b_scalar:
        raise TypeError("Trying to apply " + str(f) + " with mixed types")

    if a_scalar and not b_scalar:
        a = tf.cast(a, b.dtype)

    if b_scalar and not a_scalar:
        b = tf.cast(b, a.dtype)

    return f(a, b)


def div(a, b):
    """
    A wrapper around tf division that does more automatic casting of
    the input.
    """

    def divide(a, b):
        """Division"""
        return a / b

    return op_with_scalar_cast(a, b, divide)


def clip_eta(eta, ord, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    reduc_ind = list(xrange(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        eta = clip_by_value(eta, -eps, eps)
    elif ord == 1:
        # Implements a projection algorithm onto the l1-ball from
        # (Duchi et al. 2008) that runs in time O(d*log(d)) where d is the
        # input dimension.
        # Paper link (Duchi et al. 2008): https://dl.acm.org/citation.cfm?id=1390191

        eps = tf.cast(eps, eta.dtype)

        dim = tf.reduce_prod(tf.shape(eta)[1:])
        eta_flat = tf.reshape(eta, (-1, dim))
        abs_eta = tf.abs(eta_flat)

        if 'sort' in dir(tf):
            mu = -tf.sort(-abs_eta, axis=-1)
        else:
            # `tf.sort` is only available in TF 1.13 onwards
            mu = tf.nn.top_k(abs_eta, k=dim, sorted=True)[0]
        cumsums = tf.cumsum(mu, axis=-1)
        js = tf.cast(tf.divide(1, tf.range(1, dim + 1)), eta.dtype)
        t = tf.cast(tf.greater(mu - js * (cumsums - eps), 0), eta.dtype)

        rho = tf.argmax(t * cumsums, axis=-1)
        rho_val = tf.reduce_max(t * cumsums, axis=-1)
        theta = tf.divide(rho_val - eps, tf.cast(1 + rho, eta.dtype))

        eta_sgn = tf.sign(eta_flat)
        eta_proj = eta_sgn * tf.maximum(abs_eta - theta[:, tf.newaxis], 0)
        eta_proj = tf.reshape(eta_proj, tf.shape(eta))

        norm = tf.reduce_sum(tf.abs(eta), reduc_ind)
        eta = tf.where(tf.greater(norm, eps), eta_proj, eta)

    elif ord == 2:
        # avoid_zero_div must go inside sqrt to avoid a divide by zero
        # in the gradient through this operation
        norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                  tf.reduce_sum(tf.square(eta),
                                                reduc_ind,
                                                keepdims=True)))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        factor = tf.minimum(1., div(eps, norm))
        eta = eta * factor
    return eta


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the exponential distribution, which is not
    included in core TensorFlow.
    """
    return tf.random_gamma(shape, alpha=1, beta=1. / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the Laplace distribution, which is not
    included in core TensorFlow.
    """
    z1 = random_exponential(shape, loc, dtype=dtype, seed=seed)
    z2 = random_exponential(shape, scale, dtype=dtype, seed=seed)
    return z1 - z2


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    """
    Helper function to generate uniformly random vectors from a norm ball of
    radius epsilon.
    :param shape: Output shape of the random sample. The shape is expected to be
                  of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                  i.i.d. samples that will be drawn from a norm ball of dimension
                  `d1*d1*...*dn`.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, radius of the norm ball.
    """
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')

    if ord == np.inf:
        r = tf.random_uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:

        # For ord=1 and ord=2, we use the generic technique from
        # (Calafiore et al. 1998) to sample uniformly from a norm ball.
        # Paper link (Calafiore et al. 1998):
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
        # We first sample from the surface of the norm ball, and then scale by
        # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
        # and `d` is the dimension of the ball. In high dimensions, this is roughly
        # equivalent to sampling from the surface of the ball.

        dim = tf.reduce_prod(shape[1:])

        if ord == 1:
            x = random_laplace((shape[0], dim), loc=1.0, scale=1.0, dtype=dtype,
                               seed=seed)
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random_normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError('ord must be np.inf, 1, or 2.')

        w = tf.pow(tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
                   1.0 / tf.cast(dim, dtype))
        r = eps * tf.reshape(w * x / norm, shape)

    return r


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
    """
    A wrapper for clip_by_value that casts the clipping range if needed.
    """

    def cast_clip(clip):
        """
        Cast clipping range argument if needed.
        """
        if t.dtype in (tf.float32, tf.float64):
            if hasattr(clip, 'dtype'):
                # Convert to tf dtype in case this is a numpy dtype
                clip_dtype = tf.as_dtype(clip.dtype)
                if clip_dtype != t.dtype:
                    return tf.cast(clip, t.dtype)
        return clip

    clip_value_min = cast_clip(clip_value_min)
    clip_value_max = cast_clip(clip_value_max)

    return tf.clip_by_value(t, clip_value_min, clip_value_max, name)
