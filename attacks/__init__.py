from keras import backend as K
import tensorflow as tf
import numpy as np
import click


class Attack_FastGradientMethod(object):
    def __init__(self, eps=0.3, clip_min=0.0, clip_max=1.0):
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, X, Y):
        from .cleverhans.attacks import FastGradientMethod
        fgsm = FastGradientMethod(model, sess=K.get_session())
        fgsm_params = {'eps': self.eps, 'ord': np.inf, 'y': None, 'clip_min': self.clip_min, 'clip_max': self.clip_max}

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] FGSM Attacking %(info)s', fill_char='>',
                               empty_char='-') as bar:
            for sample_ind in bar:
                X_adv.append(fgsm.generate_np(X[sample_ind:(sample_ind + 1)], **fgsm_params))
        return np.vstack(X_adv)


class Attack_BasicIterativeMethod(object):
    def __init__(self, eps=0.3, eps_iter=0.03, nb_iter=40, clip_min=0.0, clip_max=1.0):
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, X, Y):
        from .cleverhans.attacks import BasicIterativeMethod
        bim = BasicIterativeMethod(model, sess=K.get_session())
        bim_params = {'eps': self.eps, 'eps_iter': self.eps_iter, 'nb_iter': self.nb_iter,
                      'y': None, 'ord': np.inf, 'clip_min': self.clip_min, 'clip_max': self.clip_max}

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] BIM Attacking %(info)s', fill_char='>', empty_char='-') as bar:
            for sample_ind in bar:
                X_adv.append(bim.generate_np(X[sample_ind:(sample_ind + 1)], **bim_params))
        return np.vstack(X_adv)


class Attack_ProjectedGradientDescent(object):
    def __init__(self, eps=0.3, eps_iter=0.03, nb_iter=40, random_start=True, clip_min=0.0, clip_max=1.0):
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, X, Y):
        from .madry.pgd import LinfPGDAttack, PGDModelWrapper
        x = tf.placeholder(tf.float32, shape=(None, *X.shape[1:]))
        y = tf.placeholder(tf.float32, shape=(None, Y.shape[1]))
        pgd_params = {'model': PGDModelWrapper(model, x, y), 'eps': self.eps,
                      'k': self.nb_iter, 'eps_iter': self.eps_iter, 'random_start': self.random_start}
        pgd = LinfPGDAttack(**pgd_params)

        y = np.argmax(Y, 1)
        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] PGD Attacking %(info)s', fill_char='>', empty_char='-') as bar:
            for sample_ind in bar:
                X_adv.append(pgd.perturb(X[sample_ind:(sample_ind + 1)], y[sample_ind:(sample_ind + 1)],
                                         K.get_session()))
        return np.vstack(X_adv)


class Attack_DeepFool(object):
    def __init__(self, overshoot=10.0, max_iter=50, clip_min=0.0, clip_max=1.0):
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, X, Y):
        from .lts4.deepfool import deepfool, prepare_attack
        x = tf.placeholder(tf.float32, shape=(None, *X.shape[1:]))
        f, grad_fs = prepare_attack(K.get_session(), model, x, Y.shape[1])
        deepfool_params = {'num_classes': Y.shape[1], 'overshoot': self.overshoot, 'max_iter': self.max_iter}

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] DeepFool Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for sample_ind in bar:
                X_adv.append(deepfool(X[sample_ind:sample_ind + 1, :, :, :], f, grad_fs, **deepfool_params))
        return np.vstack(X_adv)


class Attack_CarliniL2(object):
    def __init__(self, confidence=0, max_iterations=10000, learning_rate=1e-2, binary_search_steps=9,
                 initial_const=1e-3, batch_size=1, abort_early=True, targeted=True):
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.targeted = targeted

    def attack(self, model, X, Y):
        from .carlini.l2_attack import CarliniL2, wrap_to_carlini_model
        model_wrapper = wrap_to_carlini_model(model, X, Y)
        cw_params = {'confidence': self.confidence, 'targeted': self.targeted, 'learning_rate': self.learning_rate,
                     'binary_search_steps': self.binary_search_steps, 'max_iterations': self.max_iterations,
                     'abort_early': self.abort_early, 'initial_const': self.initial_const}
        attack = CarliniL2(K.get_session(), model_wrapper, **cw_params)

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] Carlini L2 Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for i in bar:
                if i % self.batch_size == 0:
                    X_sub = X[i:min(i + self.batch_size, len(X)), :]
                    Y_sub = Y[i:min(i + self.batch_size, len(X)), :]
                    X_adv.append(attack.attack(X_sub - 0.5, Y_sub) + 0.5)
        return np.vstack(X_adv)


class Attack_CarliniLi(object):
    def __init__(self, confidence=0, max_iterations=1000, learning_rate=5e-3,
                 initial_const=1e-5, largest_const=2e+1, reduce_const=False, decrease_factor=0.9, const_factor=2.0,
                 batch_size=1, abort_early=True, targeted=True):
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.largest_const = largest_const
        self.reduce_const = reduce_const
        self.decrease_factor = decrease_factor
        self.const_factor = const_factor
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.targeted = targeted

    def attack(self, model, X, Y):
        from .carlini.li_attack import CarliniLi, wrap_to_carlini_model
        model_wrapper = wrap_to_carlini_model(model, X, Y)
        cw_params = {'targeted': self.targeted, 'learning_rate': self.learning_rate,
                     'max_iterations': self.max_iterations, 'abort_early': self.abort_early,
                     'initial_const': self.initial_const, 'largest_const': self.largest_const,
                     'reduce_const': self.reduce_const, 'decrease_factor': self.decrease_factor,
                     'const_factor': self.const_factor, 'confidence': self.confidence}
        attack = CarliniLi(K.get_session(), model_wrapper, **cw_params)

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] Carlini Li Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for i in bar:
                if i % self.batch_size == 0:
                    X_sub = X[i:min(i + self.batch_size, len(X)), :]
                    Y_sub = Y[i:min(i + self.batch_size, len(X)), :]
                    X_adv.append(attack.attack(X_sub - 0.5, Y_sub) + 0.5)
        return np.vstack(X_adv)


class Attack_CarliniL0(object):
    def __init__(self, confidence=.01, max_iterations=1000, learning_rate=1e-2, independent_channels=False,
                 initial_const=1e-3, largest_const=2e6, reduce_const=False, const_factor=2.0,
                 batch_size=1, abort_early=True, targeted=True):
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.largest_const = largest_const
        self.reduce_const = reduce_const
        self.independent_channels = independent_channels
        self.const_factor = const_factor
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.targeted = targeted

    def attack(self, model, X, Y):
        from .carlini.l0_attack import CarliniL0, wrap_to_carlini_model
        model_wrapper = wrap_to_carlini_model(model, X, Y)
        cw_params = {'targeted': self.targeted, 'learning_rate': self.learning_rate,
                     'max_iterations': self.max_iterations, 'abort_early': self.abort_early,
                     'initial_const': self.initial_const, 'largest_const': self.largest_const,
                     'reduce_const': self.reduce_const, 'const_factor': self.const_factor,
                     'independent_channels': self.independent_channels, 'confidence': self.confidence}

        attack = CarliniL0(K.get_session(), model_wrapper, **cw_params)

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] Carlini L0 Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for i in bar:
                if i % self.batch_size == 0:
                    X_sub = X[i:min(i + self.batch_size, len(X)), :]
                    Y_sub = Y[i:min(i + self.batch_size, len(X)), :]
                    X_adv.append(attack.attack(X_sub - 0.5, Y_sub) + 0.5)
        return np.vstack(X_adv)


class Attack_JacobianSaliencyMapMethod(object):
    def __init__(self, theta=1.0, gamma=0.1, clip_min=0.0, clip_max=1.0):
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, X, Y):
        from .cleverhans.attacks import SaliencyMapMethod
        jsma = SaliencyMapMethod(model, sess=K.get_session())
        jsma_params = {'theta': self.theta, 'gamma': self.gamma, 'nb_classes': Y.shape[1],
                       'clip_min': self.clip_min, 'clip_max': self.clip_max, 'y_val': None,
                       'targets': tf.placeholder(tf.float32, shape=(None, Y.shape[1]))}

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True,
                               width=40, bar_template='  [%(bar)s] JSMA Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for sample_ind in bar:
                jsma_params['y_val'] = Y[[sample_ind], ]
                X_adv.append(jsma.generate_np(X[sample_ind:(sample_ind + 1)], **jsma_params))
        return np.vstack(X_adv)


class Attack_EADL1(object):
    def __init__(self, confidence=0, targeted=True, learning_rate=1e-2, binary_search_steps=9, max_iterations=10000,
                 abort_early=True, initial_const=1e-3, beta=1e-3, batch_size=1):
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
        self.batch_size = batch_size

    def attack(self, model, X, Y):
        from .ead.l1_attack import EADL1, wrap_to_ead_model
        model_wrapper = wrap_to_ead_model(model, X, Y)
        attack_params = {'confidence': self.confidence, 'targeted': self.targeted, 'learning_rate': self.learning_rate,
                         'binary_search_steps': self.binary_search_steps, 'max_iterations': self.max_iterations,
                         'abort_early': self.abort_early, 'initial_const': self.initial_const, 'beta': self.beta}

        attack = EADL1(K.get_session(), model_wrapper, **attack_params)

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] EAD L1 Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for i in bar:
                if i % self.batch_size == 0:
                    X_sub = X[i:min(i + self.batch_size, len(X)), :]
                    Y_sub = Y[i:min(i + self.batch_size, len(X)), :]
                    X_adv.append(attack.attack(X_sub - 0.5, Y_sub) + 0.5)
        return np.vstack(X_adv)


class Attack_EADEN(object):
    def __init__(self, confidence=0, targeted=True, learning_rate=1e-2, binary_search_steps=9,
                 max_iterations=10000, abort_early=True, initial_const=1e-3, beta=1e-3, batch_size=1):
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
        self.batch_size = batch_size

    def attack(self, model, X, Y):
        from .ead.en_attack import EADEN, wrap_to_ead_model
        model_wrapper = wrap_to_ead_model(model, X, Y)
        attack_params = {'confidence': self.confidence, 'targeted': self.targeted, 'learning_rate': self.learning_rate,
                         'binary_search_steps': self.binary_search_steps, 'max_iterations': self.max_iterations,
                         'abort_early': self.abort_early, 'initial_const': self.initial_const, 'beta': self.beta}

        attack = EADEN(K.get_session(), model_wrapper, **attack_params)

        X_adv = []
        with click.progressbar(range(0, len(X)), show_pos=True, width=40,
                               bar_template='  [%(bar)s] EAD EN Attacking %(info)s',
                               fill_char='>', empty_char='-') as bar:
            for i in bar:
                if i % self.batch_size == 0:
                    X_sub = X[i:min(i + self.batch_size, len(X)), :]
                    Y_sub = Y[i:min(i + self.batch_size, len(X)), :]
                    X_adv.append(attack.attack(X_sub - 0.5, Y_sub) + 0.5)
        return np.vstack(X_adv)
