from keras.models import Model
import tensorflow as tf
import numpy as np

def deepfool(image, f, grads, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for universal (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        gradients = np.asarray(grads(pert_image,I))

        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        pert_image = image + (1+overshoot)*r_tot
        pert_image = np.clip(pert_image, 0.0, 1.0)
        loop_i += 1

        # compute new label
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return pert_image


def prepare_attack(sess, model, x, nb_classes):
    f = model.predict
    model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    persisted_input = x
    persisted_output = model_logits(x)
    print('>> Compiling the gradient TensorFlow functions. This might take some time...')
    scalar_out = [tf.slice(persisted_output, [0, i], [1, 1]) for i in range(0, nb_classes)]
    dydx = [tf.gradients(scalar_out[i], [persisted_input])[0] for i in range(0, nb_classes)]

    print('>> Computing gradient function...')
    def grad_fs(image_inp, inds):
        return [sess.run(dydx[ind], feed_dict={persisted_input: image_inp}) for ind in inds]

    return f, grad_fs
