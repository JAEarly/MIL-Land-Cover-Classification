import matplotlib.pyplot as plt
import wandb

from bonfire.util import get_device
from bonfire.util import load_model_from_path, get_default_save_path
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz
from dgr_luc_interpretability import MilLucInterpretabilityStudy
from dgr_luc_models import get_model_clz
from matplotlib.gridspec import GridSpec

device = get_device()


def run():
    model_types = ['8_large', '16_medium', '24_medium', 'unet224', 'unet448']
    model_names = ['S2P 8', 'S2P 16', 'S2P 24', 'UNet 224', 'UNet 448']
    model_idxs = [2, 4, 2, 2, 2]
    n_models = len(model_types)
    study_id = 340798

    sat_img, mask_img, grid_mask_img = None, None, None
    pred_unweighted_masks = []
    pred_weighted_masks = []
    for i in range(n_models):
        # Setup for this model
        model_type = model_types[i]
        model_idx = model_idxs[i]
        model_clz = get_model_clz(model_type)
        dataset_clz = get_dataset_clz(model_type)
        model_path, _, _ = get_default_save_path(dataset_clz.name, model_clz.name, modifier=model_idx)

        # Parse wandb config and get training config for this model
        config_path = "config/dgr_luc_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.init(
            config=training_config,
            reinit=True,
        )

        # Get weighted mask for this model
        complete_dataset = dataset_clz.create_complete_dataset()
        model = load_model_from_path(device, model_clz, model_path)
        study = MilLucInterpretabilityStudy(device, complete_dataset, model, show_outputs=False)
        study_out = study.create_interpretation_from_id(study_id)
        # Get ground truth for 24 grid size
        if i == 2:
            sat_img, mask_img, grid_mask_img = study_out[0], study_out[1], study_out[2]
        pred_unweighted_mask, pred_weighted_mask = study_out[3], study_out[4]
        pred_unweighted_masks.append(pred_unweighted_mask)
        pred_weighted_masks.append(pred_weighted_mask)

    def format_axis(ax, title=None):
        ax.set_axis_off()
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title, fontsize=16)

    fig = plt.figure(figsize=(7, 7.5))
    gs = GridSpec(3, 6, figure=fig)

    # Top row - ground truth
    ax00 = fig.add_subplot(gs[0, 0:2])
    ax00.imshow(sat_img)
    format_axis(ax00, "Original Image")

    ax01 = fig.add_subplot(gs[0, 2:4])
    ax01.imshow(mask_img)
    format_axis(ax01, "True Pixel Mask")

    ax02 = fig.add_subplot(gs[0, 4:6])
    ax02.imshow(grid_mask_img)
    format_axis(ax02, "True Grid 24 Mask")

    # Middle row - S2P MIL
    for idx, mask in enumerate(pred_unweighted_masks[:3]):
        axis = fig.add_subplot(gs[1, idx*2:(idx+1)*2])
        axis.imshow(mask)
        format_axis(axis, model_names[idx])

    # Bottom row - UNet
    for idx, mask in enumerate(pred_unweighted_masks[3:]):
        axis = fig.add_subplot(gs[2, idx*2+1:(idx+1)*2+1])
        axis.imshow(mask)
        format_axis(axis, model_names[3 + idx])

    plt.tight_layout()

    save_path = "out/interpretability/paper/{:d}_comp.png".format(study_id)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.show()


if __name__ == "__main__":
    run()
