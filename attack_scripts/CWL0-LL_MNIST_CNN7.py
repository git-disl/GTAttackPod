from attacks import *
from datasets import *
from models import *
import time


if __name__ == '__main__':
    dataset = MNISTDataset()
    model = MNIST_carlini(rel_path='../')
    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model,
                                                                                                       balanced=True,
                                                                                                       num_examples=100)

    cw = Attack_CarliniL0(confidence=10, max_iterations=500)
    time_start = time.time()
    X_test_adv = cw.attack(model, X_test, Y_test_target_ll)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    # Evaluate the adversarial examples.
    print("\n---Statistics of CarliniL0 Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test_target_ll, targeted=True)
