# This file is modified based on: https://github.com/mzweilin/EvadeML-Zoo/blob/master/datasets/datasets_utils.py

from functools import reduce
from utils.math import reduce_precision_py
import numpy as np


def get_data_subset_with_systematic_attack_labels(dataset, model, balanced, num_examples):
    assert balanced and dataset.num_classes <= num_examples, '# of examples should be at least the # of classes'
    print("Loading the dataset...")
    X_test_all, Y_test_all = dataset.get_test_dataset()

    print("Evaluating the target model...")
    Y_pred_all = model.predict(X_test_all)
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on benign examples %.2f%%' % (accuracy_all * 100))
    print('Mean confidence on ground truth classes %.2f%%' % (mean_conf_all * 100))

    # select examples to attack.
    correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)  # Filter out the misclassified examples.
    if balanced:
        # Select the same number of examples for each class label.
        nb_examples_per_class = int(num_examples / Y_test_all.shape[1])
        correct_and_selected_idx = get_first_n_examples_id_each_class(Y_test_all[correct_idx], n=nb_examples_per_class)
        selected_idx = [correct_idx[i] for i in correct_and_selected_idx]
    else:
        selected_idx = correct_idx[:num_examples]

    print("Selected %d examples." % len(selected_idx))
    X_test, Y_test, Y_pred = X_test_all[selected_idx], Y_test_all[selected_idx], Y_pred_all[selected_idx]

    mean_conf_selected = calculate_mean_confidence(Y_pred, Y_test)
    accuracy_selected = calculate_accuracy(Y_pred, Y_test)  # The accuracy should be 100%.
    print('Test accuracy on selected benign examples %.2f%%' % (accuracy_selected * 100))
    print('Mean confidence on ground truth classes, selected %.2f%%\n' % (mean_conf_selected * 100))

    Y_test_target_ml = get_most_likely_class(Y_pred)
    Y_test_target_ll = get_least_likely_class(Y_pred)
    return X_test, Y_test, Y_test_target_ml, Y_test_target_ll


def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]


def get_most_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argsort(Y_pred, axis=1)[:, -2]
    return np.eye(num_classes)[Y_target_labels]


def get_first_n_examples_id_each_class(Y_test, n=1):
    """
    Only return the classes with samples.
    """
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)

    selected_idx = []
    for i in range(num_classes):
        loc = np.where(Y_test_labels == i)[0]
        if len(loc) > 0:
            selected_idx.append(list(loc[:n]))

    selected_idx = reduce(lambda x, y: x + y, zip(*selected_idx))

    return np.array(selected_idx)


def get_first_example_id_each_class(Y_test):
    return get_first_n_examples_id_each_class(Y_test, n=1)


def get_correct_prediction_idx(Y_pred, Y_label):
    """
    Get the index of the correct predicted samples.
    :param Y_pred: softmax output, probability matrix.
    :param Y_label: groundtruth classes in shape (#samples, #classes)
    :return: the index of samples being corrected predicted.
    """
    pred_classes = np.argmax(Y_pred, axis=1)
    labels_classes = np.argmax(Y_label, axis=1)

    return np.where(pred_classes == labels_classes)[0]


def calculate_mean_confidence(Y_pred, Y_target):
    """
    Calculate the mean confidence on target classes.
    :param Y_pred: softmax output
    :param Y_target: target classes in shape (#samples, #classes)
    :return: the mean confidence.
    """
    assert len(Y_pred) == len(Y_target)
    confidence = np.multiply(Y_pred, Y_target)
    confidence = np.max(confidence, axis=1)

    mean_confidence = np.mean(confidence)

    return mean_confidence


def get_match_pred_vec(Y_pred, Y_label):
    assert len(Y_pred) == len(Y_label)
    Y_pred_class = np.argmax(Y_pred, axis=1)
    Y_label_class = np.argmax(Y_label, axis=1)
    return Y_pred_class == Y_label_class


def calculate_accuracy(Y_pred, Y_label):
    match_pred_vec = get_match_pred_vec(Y_pred, Y_label)
    accuracy = np.sum(match_pred_vec) / float(len(Y_label))
    return accuracy


def calculate_mean_distance(X1, X2):
    img_size = X1.shape[1] * X1.shape[2]
    nb_channels = X1.shape[3]

    mean_l2_dist = float(np.mean([np.sum((X1[i] - X2[i]) ** 2) ** .5 for i in range(len(X1))]))
    mean_li_dist = float(np.mean([np.max(np.abs(X1[i] - X2[i])) for i in range(len(X1))]))
    diff_channel_list = np.split(X1 - X2 != 0, nb_channels, axis=3)
    l0_channel_dependent_list = np.sum(reduce(lambda x, y: x | y, diff_channel_list), axis=(1, 2, 3))
    mean_l0_dist = np.mean(l0_channel_dependent_list) / img_size

    return mean_l2_dist, mean_li_dist, mean_l0_dist


def evaluate_adversarial_examples(X_test, Y_test, X_test_adv, Y_test_target, Y_test_adv_pred, targeted):
    X_test_adv = reduce_precision_py(X_test_adv, 256)
    misclassification_rate = 1 - calculate_accuracy(Y_test_adv_pred, Y_test)
    success_rate = calculate_accuracy(Y_test_adv_pred, Y_test_target)
    success_idx = get_match_pred_vec(Y_test_adv_pred, Y_test_target)

    if targeted is False:
        success_rate = 1 - success_rate
        success_idx = np.logical_not(success_idx)

    # Calculate the mean confidence of the successful adversarial examples.
    mean_conf = calculate_mean_confidence(Y_test_adv_pred[success_idx], Y_test_target[success_idx])
    if targeted is False:
        mean_conf = 1 - mean_conf

    mean_l2_dist, mean_li_dist, mean_l0_dist = calculate_mean_distance(X_test[success_idx], X_test_adv[success_idx])
    print("Success rate: %.2f%%, "
          "Misclassification rate: %.2f%%, "
          "Mean confidence: %.2f%%" % (success_rate * 100, misclassification_rate * 100, mean_conf * 100))
    print("Li dist: %.4f, L2 dist: %.4f, L0 dist: %.1f%%" % (mean_li_dist, mean_l2_dist, mean_l0_dist * 100))
    rec = {'success_rate': success_rate,
           'mean_confidence': mean_conf,
           'mean_l2_dist': mean_l2_dist,
           'mean_li_dist': mean_li_dist,
           'mean_l0_dist': mean_l0_dist}
    return rec
