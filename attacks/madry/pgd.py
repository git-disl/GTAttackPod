from keras.models import Model
import numpy as np
import tensorflow as tf


class LinfPGDAttack:
    def __init__(self, model, eps, k, eps_iter, random_start):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.eps = eps
        self.k = k
        self.eps_iter = eps_iter
        self.rand = random_start
        self.grad = tf.gradients(model.xent, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.eps, self.eps, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})

            x += self.eps_iter * np.sign(grad)

            x = np.clip(x, x_nat - self.eps, x_nat + self.eps)
            x = np.clip(x, 0, 1)  # ensure valid pixel range

        return x


class PGDModelWrapper:
    def __init__(self, keras_model, x, y):
        model_logits = Model(inputs=keras_model.layers[0].input, outputs=keras_model.layers[-2].output)
        self.x_input = x
        self.y_input = tf.argmax(y, 1)
        self.pre_softmax = model_logits(x)
        self.xent = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                 logits=self.pre_softmax))
        self.y_pred = tf.argmax(self.pre_softmax, 1)
