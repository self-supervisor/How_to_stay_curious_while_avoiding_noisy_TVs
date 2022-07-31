import numpy as np
import pytest
import torch

from noisy_mnist_aleatoric_uncertainty_for_poster import *


@pytest.fixture(scope="module", params=["train", "test"])
def noisy_mnist_env(request):

    mnist_env = NoisyMnistEnv(request.param, 0, 2)
    return mnist_env


@pytest.fixture(scope="module", params=["mse", "aleatoric"])
def noisy_mnist_experiment(request):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_env_train = NoisyMnistEnv("train", 0, 2)
    mnist_env_test_zeros = NoisyMnistEnv("test", 0, 2)
    mnist_env_test_ones = NoisyMnistEnv("test", 0, 2)

    if request.param == "mse":
        model = Net()
        experiment = NoisyMNISTExperimentRun(
            repeats=1,
            training_steps=3000,
            checkpoint_loss=100,
            lr=0.001,
            model=model,
            mnist_env_train=mnist_env_train,
            mnist_env_test_zeros=mnist_env_test_zeros,
            mnist_env_test_ones=mnist_env_test_ones,
            device=device,
        )
    elif request.param == "aleatoric":
        model = AleatoricNet()
        experiment = NoisyMNISTExperimentRunAMA(
            repeats=1,
            training_steps=3000,
            checkpoint_loss=100,
            lr=0.001,
            model=model,
            mnist_env_train=mnist_env_train,
            mnist_env_test_zeros=mnist_env_test_zeros,
            mnist_env_test_ones=mnist_env_test_ones,
            device=device,
        )
    return experiment


def check_count_of_classes(x_arr, y_arr):
    same = 0
    not_same = 0
    for i, _ in enumerate(x_arr):
        if np.array_equal(x_arr[i], y_arr[i]):
            same += 1
        else:
            not_same += 1
    return same, not_same


def test_mnist_env_step(noisy_mnist_env):
    import math

    x_arr, y_arr = noisy_mnist_env.step()
    assert x_arr.shape == y_arr.shape  # make sure batch shapes make sense
    for i, _ in enumerate(x_arr):  # check batch is completely filled
        assert np.array_equal(x_arr[i], np.zeros((1, 28 * 28))) == False
        assert np.array_equal(y_arr[i], np.zeros((1, 28 * 28))) == False
        assert np.array_equal(np.zeros((1, 28 * 28)), np.zeros((1, 28 * 28))) == True
    same = 0
    not_same = 0
    for _ in range(
        1000
    ):  # check roughly half are deterministic transitions, half aren't
        x_arr, y_arr = noisy_mnist_env.step()
        same_sample, not_same_sample = check_count_of_classes(x_arr, y_arr)
        same += same_sample
        not_same += not_same_sample
    print("same", same)
    print("not same", not_same)
    assert math.isclose(same, not_same, rel_tol=0.2)


def test_mnist_env_random_sample_of_number(noisy_mnist_env):
    """
    This test is a qualitative visual test, look in test images
    and make sure the number title is the same as the number
    """
    import matplotlib.pyplot as plt
    import os
    import shutil

    if os.path.isdir("unit_test_images"):
        shutil.rmtree("unit_test_images")
    os.mkdir("unit_test_images")

    for number in range(0, 10):
        digit = noisy_mnist_env.get_random_sample_of_number(number)
        plt.imshow(np.array(digit).reshape(28, 28))
        plt.title(str(number))
        plt.savefig("unit_test_images/" + str(number) + ".png")


def test_run_experiment(noisy_mnist_experiment):
    noisy_mnist_experiment.run_experiment()


def test_get_batch(noisy_mnist_experiment):
    envs = [
        noisy_mnist_experiment.env_train,
        noisy_mnist_experiment.env_test_zeros,
        noisy_mnist_experiment.env_test_ones,
    ]
    for an_env in envs:
        data, target = noisy_mnist_experiment.get_batch(an_env)
        assertions_for_generated_data(data)
        assertions_for_generated_data(target)


def assertions_for_generated_data(input_tensor):
    assert input_tensor.type() == "torch.cuda.FloatTensor"
    assert input_tensor.max() <= 1.0
    assert input_tensor.min() >= 0.0
    assert torch.all(torch.eq(input_tensor, torch.zeros_like(input_tensor))) == False
    assert (
        torch.all(
            torch.eq(torch.zeros_like(input_tensor), torch.zeros_like(input_tensor))
        )
        == True
    )
    assert batch_is_different(input_tensor) == True
    assert batch_is_different(torch.zeros_like(input_tensor)) == False


def batch_is_different(input_tensor):
    duplicate_tensors = 0
    for i, data_point_i in enumerate(input_tensor):
        for j, data_point_j in enumerate(input_tensor):
            if i != j:
                if torch.all(torch.eq(data_point_i, data_point_j)):
                    duplicate_tensors += 1
    if duplicate_tensors != (len(input_tensor) - 1) * len(input_tensor):
        return True
    return False


def test_train_step(noisy_mnist_experiment):
    import copy

    model_copy = copy.deepcopy(noisy_mnist_experiment.model)
    loss_buffer_copy = copy.deepcopy(noisy_mnist_experiment.loss_buffer)
    noisy_mnist_experiment.train_step(1)

    assert_model_gets_updated(model_copy, noisy_mnist_experiment.model)

    assert len(loss_buffer_copy) == 0
    assert len(noisy_mnist_experiment.loss_buffer) > len(loss_buffer_copy)
    noisy_mnist_experiment.train_step(noisy_mnist_experiment.checkpoint_loss - 1)
    assert len(noisy_mnist_experiment.loss_buffer) == 0


def assert_model_gets_updated(old_model, updated_model):
    params = get_params_from_model(updated_model)
    copy_params = get_params_from_model(old_model)

    for i, _ in enumerate(params):
        assert torch.all(torch.eq(copy_params[i], params[i])) == False
        assert torch.all(torch.eq(copy_params[i], copy_params[i])) == True


def get_params_from_model(a_model):
    params = []
    for name, param in a_model.named_parameters():
        params.append(param)
    return param


def test_eval_step(noisy_mnist_experiment):
    import copy

    loss_buffer_1_copy = copy.deepcopy(noisy_mnist_experiment.loss_buffer_1)
    assert len(loss_buffer_1_copy) == 0

    noisy_mnist_experiment.eval_step("ones", 0)
    assert len(noisy_mnist_experiment.loss_buffer_1) > len(loss_buffer_1_copy)
    noisy_mnist_experiment.eval_step("ones", noisy_mnist_experiment.checkpoint_loss - 1)
    assert len(noisy_mnist_experiment.loss_buffer_1) == 0

    loss_buffer_0_copy = copy.deepcopy(noisy_mnist_experiment.loss_buffer_0)
    assert len(loss_buffer_0_copy) == 0

    noisy_mnist_experiment.eval_step("zeros", 0)
    assert len(noisy_mnist_experiment.loss_buffer_0) > len(loss_buffer_0_copy)
    noisy_mnist_experiment.eval_step(
        "zeros", noisy_mnist_experiment.checkpoint_loss - 1
    )
    assert len(noisy_mnist_experiment.loss_buffer_0) == 0
