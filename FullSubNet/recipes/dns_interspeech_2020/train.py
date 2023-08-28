import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import toml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(
    os.path.abspath(os.path.join(__file__, "..", "..", ".."))
)  # without installation, add /path/to/Audio-ZEN
import audio_zen.loss as loss
from audio_zen.utils import initialize_module


def entry(rank, config, resume, only_validation):
    torch.manual_seed(config["meta"]["seed"])  # For both CPU and GPU
    np.random.seed(config["meta"]["seed"])
    random.seed(config["meta"]["seed"])
    torch.cuda.set_device(rank)

    # Initialize the process group
    # The environment variables necessary to initialize a Torch process group are provided to you by this module,
    # and no need for you to pass ``RANK`` manually.
    torch.distributed.init_process_group(backend="nccl")
    print(f"Process {rank + 1} initialized.")

    # The DistributedSampler will split the dataset into the several cross-process parts.
    # On the contrary, setting "Sampler=None, shuffle=True", each GPU will get all data in the whole dataset.
    train_dataset = initialize_module(
        config["train_dataset"]["path"], args=config["train_dataset"]["args"]
    )
    sampler = DistributedSampler(dataset=train_dataset, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        shuffle=False,
        **config["train_dataset"]["dataloader"],
    )

    valid_dataloader = DataLoader(
        dataset=initialize_module(
            config["validation_dataset"]["path"], args=config["validation_dataset"]["args"]
        ),
        num_workers=0,
        batch_size=1,
    )

    model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
    )

    loss_function = getattr(loss, config["loss_function"]["name"])(
        **config["loss_function"]["args"]
    )

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    trainer = trainer_class(
        dist=dist,
        rank=rank,
        config=config,
        resume=resume,
        only_validation=only_validation,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FullSubNet")
    parser.add_argument(
        "-C", "--configuration", required=True, type=str, help="Configuration (*.toml)."
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-V",
        "--only_validation",
        action="store_true",
        help="Only run validation, which is used for debugging.",
    )
    parser.add_argument(
        "-P", "--preloaded_model_path", type=str, help="Path of the *.pth file of a model."
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    if args.preloaded_model_path:
        assert not args.resume, "The 'resume' conflicts with the 'preloaded_model_path'."

    config_path = Path(args.configuration).expanduser().absolute()
    configuration = toml.load(config_path.as_posix())

    # append the parent dir of the config path to python's context
    # /path/to/recipes/dns_interspeech_2020/exp/'
    sys.path.append(config_path.parent.as_posix())

    configuration["meta"]["experiment_name"], _ = os.path.splitext(
        os.path.basename(args.configuration)
    )
    configuration["meta"]["config_path"] = args.configuration
    configuration["meta"]["preloaded_model_path"] = args.preloaded_model_path

    entry(local_rank, configuration, args.resume, args.only_validation)
