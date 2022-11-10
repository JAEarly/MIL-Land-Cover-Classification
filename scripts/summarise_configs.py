import wandb

from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_list
from dgr_luc_models import get_n_params
from texttable import Texttable

import time

from latextable import draw_latex


def summarise_configs():
    wandb.init()

    rows = [['Configuration', 'Grid Size', 'Cell Size', 'Patch Size', 'Eff. Resolution', 'Scale', '\\# Params']]
    for dataset in get_dataset_list():
        patch_details = dataset.patch_details
        model_type = dataset.model_type

        config_path = "config/dgr_luc_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)

        row = [
            _format_model_type(model_type),
            "{:d} x {:d}".format(patch_details.grid_size, patch_details.grid_size),
            "{:d} x {:d} px".format(patch_details.cell_size, patch_details.cell_size),
            "{:d} x {:d} px".format(patch_details.patch_size, patch_details.patch_size),
            "{:d} x {:d} px".format(patch_details.effective_resolution, patch_details.effective_resolution),
            "{:.1f}\\%".format(patch_details.scale * 100),
            "{:s}".format(_format_n_params(get_n_params(model_type))),
        ]

        rows.append(row)

    table = Texttable()
    table.add_rows(rows)
    table.set_cols_align(['l'] * 7)
    table.set_max_width(0)
    print(table.draw())

    print('\n')

    print(draw_latex(table, use_booktabs=True))

    # For wandb
    time.sleep(3)


def _format_model_type(model_type):
    if model_type == 'resnet':
        return 'ResNet18'
    elif 'unet' in model_type:
        return 'UNet {:s}'.format(model_type[-3:])
    grid_size, patch_size = model_type.split('_')
    return 'MIL {:s} {:s}'.format(patch_size.title(), grid_size)


def _format_n_params(n_params):
    if n_params >= 1e6:
        return "{:.1f}M".format(n_params/1e6)
    else:
        return "{:.0f}K".format(n_params/1e3)


if __name__ == "__main__":
    summarise_configs()
