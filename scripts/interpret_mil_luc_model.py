import argparse

import wandb

from bonfire.util import get_device
from bonfire.util import load_model_from_path, get_default_save_path
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, get_model_type_list
from dgr_luc_interpretability import MilLucInterpretabilityStudy
from dgr_luc_models import get_model_clz

device = get_device()
model_type_choices = get_model_type_list()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC interpretability script.')
    parser.add_argument('model_type', choices=model_type_choices, help="Type of model to interpret.")
    parser.add_argument('task', choices=['reconstruct', 'sample', 'specific'], help='The task to perform.')
    parser.add_argument('-s', '--show_outputs', action='store_true',
                        help="Whether or not to show the interpretability outputs (they're always saved).")
    args = parser.parse_args()
    return args.model_type, args.task, args.show_outputs


def run():
    model_type, task, show_outputs = parse_args()

    # Best idx by model type
    if model_type == '24_medium':
        model_idx = 2
    elif model_type == '16_medium':
        model_idx = 4
    elif model_type == '8_large':
        model_idx = 2
    elif model_type == 'unet224':
        model_idx = 2
    elif model_type == 'unet448':
        model_idx = 2
    else:
        raise NotImplementedError

    model_clz = get_model_clz(model_type)
    dataset_clz = get_dataset_clz(model_type)
    model_path, _, _ = get_default_save_path(dataset_clz.name, model_clz.name, modifier=model_idx)

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_type)
    wandb.init(
        config=training_config,
    )

    complete_dataset = dataset_clz.create_complete_dataset()
    model = load_model_from_path(device, model_clz, model_path)
    study = MilLucInterpretabilityStudy(device, complete_dataset, model, show_outputs)

    if task == 'reconstruct':
        study.create_reconstructions()
    elif task == 'sample':
        study.sample_interpretations()
    elif task == 'specific':
        study.create_interpretation_from_id(941237)
    else:
        raise NotImplementedError('Task not implemented: {:s}.'.format(task))


if __name__ == "__main__":
    run()
