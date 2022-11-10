import argparse

from codecarbon import OfflineEmissionsTracker

from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, get_model_type_list
from dgr_luc_models import get_model_clz

device = get_device()
model_type_choices = get_model_type_list()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('model', choices=model_type_choices, help="Type of model to train.")
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    parser.add_argument('-t', '--track_emissions', action='store_true',
                        help='Whether or not to track emissions using CodeCarbon.')
    args = parser.parse_args()
    return args.model, args.n_repeats, args.track_emissions


def run_training():
    model_type, n_repeats, track_emissions = parse_args()

    tracker = None
    if track_emissions:
        tracker = OfflineEmissionsTracker(country_iso_code="GBR", project_name="Train_MIL_LUC_Model",
                                          output_dir="out/emissions", log_level='error')
        tracker.start()

    model_clz = get_model_clz(model_type)
    dataset_clz = get_dataset_clz(model_type)

    project_name = "Train_MIL_LUC"
    group_name = "Train_{:s}".format(model_type)
    if 'resnet' in model_type or 'unet' in model_type:
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, project_name=project_name,
                                           dataloader_func=create_normal_dataloader, group_name=group_name)
    else:
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz,
                                           project_name=project_name, group_name=group_name)

    dataset_name = dataset_clz.name

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_type)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:} {:} and dataset {:}'.format(model_type, model_clz, dataset_clz))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(training_config, n_repeats=n_repeats)

    if tracker:
        tracker.stop()


if __name__ == "__main__":
    run_training()
