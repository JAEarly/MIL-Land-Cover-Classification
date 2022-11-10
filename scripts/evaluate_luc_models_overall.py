import argparse

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm

from bonfire.train.metrics import output_results, IoUMetric
from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device, load_model
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, get_model_type_list, get_patch_details, make_binary_mask
from dgr_luc_models import get_model_clz

device = get_device()
all_models = get_model_type_list()
model_type_choices = all_models + ['all']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('model_types', choices=model_type_choices, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.model_types, args.n_repeats


def run_evaluation():
    wandb.init()

    model_types, n_repeats = parse_args()

    if model_types == ['all']:
        model_types = all_models

    # print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for models: {:}'.format(model_types))

    bag_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    grid_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    orig_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    for model_idx, model_type in enumerate(model_types):
        print('Evaluating {:s}'.format(model_type))

        model_clz = get_model_clz(model_type)
        dataset_clz = get_dataset_clz(model_type)

        config_path = "config/dgr_luc_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)

        if 'resnet' in model_type or 'unet' in model_type:
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, dataloader_func=create_normal_dataloader)
        else:
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)

        model_results = evaluate(model_type, n_repeats, trainer)
        model_bag_results, model_grid_seg_results, model_orig_seg_results = model_results
        bag_results[model_idx, :, :] = model_bag_results
        grid_seg_results[model_idx, :, :] = model_grid_seg_results
        orig_seg_results[model_idx, :, :] = model_orig_seg_results
    print('\nBag Results')
    output_results(model_types, bag_results, sort=False)
    print('\nGrid Segmentation Results')
    output_results(model_types, grid_seg_results, sort=False)
    print('\nHigh-Res Segmentation Results')
    output_results(model_types, orig_seg_results, sort=False)


def evaluate(model_type, n_repeats, trainer, random_state=5):
    bag_results_arr = np.empty((n_repeats, 3), dtype=object)
    grid_seg_results_arr = np.empty((n_repeats, 3), dtype=object)
    orig_seg_results_arr = np.empty((n_repeats, 3), dtype=object)

    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, True, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(device, trainer.dataset_clz.name, trainer.model_clz, modifier=r)

        results_list = eval_complete(model_type, trainer.metric_clz, model,
                                     train_dataloader, val_dataloader, test_dataloader, verbose=False)

        train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res = results_list
        bag_results_arr[r, :] = [train_bag_res[0], val_bag_res[0], test_bag_res[0]]
        grid_seg_results_arr[r, :] = [train_inst_res[0], val_inst_res[0], test_inst_res[0]]
        orig_seg_results_arr[r, :] = [train_inst_res[1], val_inst_res[1], test_inst_res[1]]

        r += 1
        if r == n_repeats:
            break

    return bag_results_arr, grid_seg_results_arr, orig_seg_results_arr


def eval_complete(model_type, bag_metric, model, train_dataloader, val_dataloader, test_dataloader, verbose=False):
    train_bag_res, train_inst_res = eval_model(model_type, bag_metric, model, train_dataloader, verbose=verbose)
    val_bag_res, val_inst_res = eval_model(model_type, bag_metric, model, val_dataloader, verbose=verbose)
    test_bag_res, test_inst_res = eval_model(model_type, bag_metric, model, test_dataloader, verbose=verbose)
    return train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res


def eval_model(model_type, bag_metric, model, dataloader, verbose=False):
    # Iterate through data loader and gather preds and targets
    all_preds = []
    all_targets = []
    all_instance_preds = []
    all_instance_targets = []
    all_mask_paths = []
    labels = list(range(model.n_classes))
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Getting model predictions', leave=False):
            bags, targets, instance_targets, mask_path = data[0], data[1], data[2], data[3]
            bag_pred, instance_pred = model.forward_verbose(bags)
            all_preds.append(bag_pred.cpu())
            all_targets.append(targets.cpu())
            all_mask_paths.append(mask_path[0])
            instance_pred = instance_pred[0]
            if instance_pred is not None:
                all_instance_preds.append(instance_pred.squeeze().cpu())
            all_instance_targets.append(instance_targets.squeeze().cpu())

    # Calculate bag results
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    bag_results = [bag_metric.calculate_metric(all_preds, all_targets, labels)]
    if verbose:
        for bag_result in bag_results:
            bag_result.out()

    # Calculate instance results
    grid_results = IoUMetric(torch.nan, torch.nan, None)
    seg_results = IoUMetric(torch.nan, torch.nan, None)
    if model_type != 'resnet':
        all_instance_preds = torch.stack(all_instance_preds)
        all_instance_targets = torch.stack(all_instance_targets)
        if 'unet' in model_type:
            # No evaluation for grid segmentation
            seg_results = evaluate_iou_segmentation(dataloader.dataset, all_instance_preds, labels, all_mask_paths)
        elif model_type != 'resnet':
            # Wrangle targets to grid shape
            #  Swap class and instance axes
            #  Reshape to match the image grid
            patch_details = get_patch_details(model_type)
            grid_targets = all_instance_targets\
                .swapaxes(1, 2)\
                .reshape(-1, len(labels), patch_details.grid_size, patch_details.grid_size)
            grid_results = evaluate_iou_grid(all_instance_preds, grid_targets, labels)
            seg_results = evaluate_iou_segmentation(dataloader.dataset, all_instance_preds, labels, all_mask_paths)

    instance_results = [grid_results, seg_results]
    if verbose:
        for instance_result in instance_results:
            instance_result.out()

    return bag_results, instance_results


def evaluate_iou_grid(grid_predictions, grid_targets, labels):
    # Evaluate IoU on grid preds and targets
    grid_clz_predictions = torch.argmax(grid_predictions, dim=1).long()
    grid_clz_targets = torch.argmax(grid_targets, dim=1).long()
    return IoUMetric.calculate_metric(grid_clz_predictions, grid_clz_targets, labels)


def evaluate_iou_segmentation(dataset, all_grid_predictions, labels, mask_paths):
    """
    Evaluate IoU against original high res segmented images by scaling up the low resolution grid predictions
    """
    all_grid_clz_predictions = torch.argmax(all_grid_predictions, dim=1).long()

    # Compute IoU by first calculating the unions and intersections for every image, then doing a final computation
    # Storing all the predicted masks and true masks is too expensive
    all_conf_mats = []
    for idx, grid_clz_predictions in tqdm(enumerate(all_grid_clz_predictions), desc='Computing high res mIOU',
                                          leave=False, total=len(all_grid_clz_predictions)):
        # Load true mask image to compare to
        mask_img = Image.open(mask_paths[idx])

        # Threshold image to overcome colour variations
        #  PIL img -> np arr -> torch tensor
        mask_binary = make_binary_mask(torch.as_tensor(np.array(mask_img)))
        # Convert thresholded tensor back to PIL image
        #  torch tensor -> np arr -> PIL img
        mask_img = Image.fromarray(mask_binary.numpy() * 255)

        # Convert coloured mask image to clz mask tensor using the clz palette
        clz_palette = dataset.create_clz_palette()
        p_img = Image.new('P', (2448, 2448))
        p_img.putpalette(clz_palette)
        mask_img = mask_img.quantize(palette=p_img, dither=0)
        mask_clz_tensor = torch.as_tensor(np.array(mask_img))

        # Scale up grid predictions to same size as original image
        #  Have to double unsqueeze to add batch and channel dimensions so interpolation acts in the correct dimensions
        pred_clz_tensor = F.interpolate(grid_clz_predictions.float().unsqueeze(0).unsqueeze(0),
                                        size=(2448, 2448), mode='nearest-exact')
        pred_clz_tensor = pred_clz_tensor.squeeze().long()

        # Compute intersection and union for this bag (used to calculate an overall IOU later)
        _, _, conf_mat = IoUMetric.intersection_over_union(mask_clz_tensor, pred_clz_tensor, len(labels))
        all_conf_mats.append(conf_mat)

    # Compute the final IoU score
    mean_iou, clz_iou, conf_mat = IoUMetric.calculate_from_cumulative(all_conf_mats)
    met = IoUMetric(mean_iou, clz_iou, conf_mat)
    return met


if __name__ == "__main__":
    run_evaluation()
