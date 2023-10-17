import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from model import Net
from perfedavg import PerFedAvgClient
import torch


# from PerfedAvg import PerFedAvg

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    save_path = HydraConfig.get().runtime.output_dir

    print("""Enter the type of the dataset you want to use:
            1. MNIST
            2. CIFAR-10
            3. CIFAR-100""")
    which = int(input())

    action = int(input("""Enter the type of operation
                       1. FedAvg
                       2. Per-FedAvg"""))

    if which == 1:
        classes_channels = 1
        classes_class = 10
    elif which == 2:
        classes_channels = 3
        classes_class = 10
    elif which ==  3:
        classes_channels = 3
        classes_class = 100

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, which)

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes, classes_channels, classes_class)

    if action == 1:
        strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.0,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=15,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, classes_channels, classes_class, which),
    )
    elif action == 2:
        strategy = PerFedAvgClient(
            alpha=cfg.alpha,  # Set your desired alpha value
            beta=cfg.beta,   # Set your desired beta value 
            global_model=Net,  # You should set this to the initial global model
            criterion=torch.nn.CrossEntropyLoss(),  # Change this to the appropriate loss function
            batch_size=cfg.batch_size,  # Set your batch size
            dataset=which,  # Set the dataset ID (1 for MNIST, 2 for CIFAR-10, 3 for CIFAR-100)
            local_epochs=4,  # Set the number of local training epochs
            valset_ratio=0.1,  # Set your validation set ratio
            trainloader=trainloaders,  # Set your trainloader
            valloader=validationloaders,  # Set your validation loader
        )
        

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()


