from collections import OrderedDict
from omegaconf import DictConfig
import torch
import json  # Add this import for working with JSON
from model import Net, test

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader, classes_channels, classes_class, which:int):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes, classes_channels, classes_class)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, testloader, device)

        # Create a dictionary to store loss and accuracy
        evaluation_results = {
            "loss": loss,
            "accuracy": accuracy
        }

        # Save the evaluation results to a JSON file
        if which == 1:
            with open("evaluation_results_MNIST.json", "a") as json_file:
                json.dump(evaluation_results, json_file)
                json_file.write("\n")
        elif which == 2:
            with open("evaluation_results_CIFAR10.json", "a") as json_file:
                json.dump(evaluation_results, json_file)
                json_file.write("\n")
        elif which == 3:
            with open("evaluation_results_CIFAR100.json", "a") as json_file:
                json.dump(evaluation_results, json_file)
                json_file.write("\n")
            

        return loss, {"accuracy": accuracy}

    return evaluate_fn
