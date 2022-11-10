import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from overrides import overrides
from sklearn.model_selection import KFold, train_test_split
from torchvision import transforms
from tqdm import tqdm

from bonfire.data.mil_dataset import MilDataset
from bonfire.train.metrics import RegressionMetric, output_regression_results


RAW_DATA_DIR = 'data/DeepGlobeLUC/raw'
PATCH_DATA_CSV_FMT = 'data/DeepGlobeLUC/patch_{:d}_data.csv'
RECONSTRUCTION_DATA_DIR_FMT = 'data/DeepGlobeLUC/reconstruction_{:d}_{:d}'
TARGET_OUT_PATH = 'data/DeepGlobeLUC/targets.csv'
CLASS_DIST_PATH = 'data/DeepGlobeLUC/class_distribution.csv'


class PatchDetails:

    def __init__(self, grid_size, patch_size, orig_img_size=2448):
        # Size of (square) grid to apply over the original image
        self.grid_size = grid_size
        # Size to reduce each cell of the grid to
        self.patch_size = patch_size
        # Size of original image
        self.orig_img_size = orig_img_size
        # Size of each cell (according to original image size and grid size)
        self.cell_size = self._calculate_cell_size()
        # Total number of patches that will be extracted from each image (size of grid squared)
        self.num_patches = self._calculate_num_patches()
        # Effective resolution after extracting and resizing cells
        self.effective_resolution = self._calculate_effective_resolution()
        # Scale of effective resolution compared to original resolution
        self.scale = self._calculate_scale()

    def _calculate_cell_size(self):
        if self.orig_img_size % self.grid_size != 0:
            raise ValueError("Invalid grid size. Must be a factor of {:d}".format(self.orig_img_size))
        return self.orig_img_size // self.grid_size

    def _calculate_num_patches(self):
        return int(self.grid_size ** 2)

    def _calculate_effective_resolution(self):
        return self.grid_size * self.patch_size

    def _calculate_scale(self):
        return (self.effective_resolution ** 2) / (self.orig_img_size ** 2)


def get_patch_details(model_type):
    if model_type == "8_small":
        return PatchDetails(8, 28)
    elif model_type == "8_medium":
        return PatchDetails(8, 56)
    elif model_type == "8_large":
        return PatchDetails(8, 102)
    elif model_type == "16_small":
        return PatchDetails(16, 28)
    elif model_type == "16_medium":
        return PatchDetails(16, 56)
    elif model_type == "16_large":
        return PatchDetails(16, 102)
    elif model_type == "24_small":
        return PatchDetails(24, 28)
    elif model_type == "24_medium":
        return PatchDetails(24, 56)
    elif model_type == "24_large":
        return PatchDetails(24, 102)
    elif model_type == "resnet":
        return PatchDetails(1, 224)
    elif model_type == "unet224":
        return PatchDetails(1, 224)
    elif model_type == "unet448":
        return PatchDetails(1, 448)
    else:
        raise ValueError("No patch details registered for model type {:s}".format(model_type))


def get_dataset_list():
    return [DgrLucDatasetResNet, DgrLucDatasetUNet224, DgrLucDatasetUNet448,
            DgrLucDataset8Small, DgrLucDataset8Medium, DgrLucDataset8Large,
            DgrLucDataset16Small, DgrLucDataset16Medium, DgrLucDataset16Large,
            DgrLucDataset24Small, DgrLucDataset24Medium, DgrLucDataset24Large]


def get_model_type_list():
    return [dataset_clz.model_type for dataset_clz in get_dataset_list()]


def get_dataset_clz(model_type):
    for dataset_clz in get_dataset_list():
        if dataset_clz.model_type == model_type:
            return dataset_clz


def setup(model_type):
    metadata_df = _load_metadata_df()
    _setup_patch_csv(metadata_df, model_type)
    # _calculate_dataset_normalisation(metadata_df)
    # _visualise_data(metadata_df)
    # _generate_per_class_coverage(metadata_df)
    # _plot_per_class_coverage()
    # _baseline_performance()


def _load_metadata_df():
    """
    Create a dataframe containing the metadata for each image (loaded from metadata.csv).
    """
    print('Loading image metadata')
    # Read data from csv
    metadata_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'metadata.csv'))
    # Discard anything that isn't train split (as other splits don't have segmentation ground truths
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    # Drop split column as we don't need it
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    # Update paths to images to be relative to project base
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    print(' Found {:d} images'.format(len(metadata_df)))
    return metadata_df


def _load_class_dict_df():
    print('Loading class data')
    class_dict_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'class_dict.csv'))
    class_dict_df[['r', 'g', 'b']] = class_dict_df[['r', 'g', 'b']].apply(lambda v: v // 255)
    class_dict_df['target'] = list(range(len(class_dict_df)))
    print(' Found {:d} classes'.format(len(class_dict_df)))
    return class_dict_df


def _setup_patch_csv(metadata_df, model_type):
    """
    Extract patches from the training images.
    :param metadata_df: Dataframe of image metadata.
    """
    patch_details = get_patch_details(model_type)

    print('Setting up patch csv')
    print(' Cell size: {:d}'.format(patch_details.cell_size))
    print(' Patch size: {:d}'.format(patch_details.patch_size))
    print(' {:d} patches per image'.format(patch_details.num_patches))
    print(' {:d} effective new size'.format(patch_details.effective_resolution))

    all_patch_data = []
    # Loop over all the images
    for i in tqdm(range(len(metadata_df)), desc='Calculating patch targets', leave=False):
        # Load mask data
        image_id = metadata_df['image_id'][i]
        mask_path = metadata_df['mask_path'][i]
        mask_img = Image.open(mask_path)
        mask_img_arr = np.array(mask_img)

        # Iterate through each cell in the grid
        n_x = int(mask_img_arr.shape[0]/patch_details.cell_size)
        n_y = int(mask_img_arr.shape[1]/patch_details.cell_size)
        for i_x in range(n_x):
            for i_y in range(n_y):
                # Extract patch from original image
                p_x = i_x * patch_details.cell_size
                p_y = i_y * patch_details.cell_size

                # Extract mask patch from original mask
                patch_mask_arr = mask_img_arr[p_x:p_x+patch_details.cell_size, p_y:p_y+patch_details.cell_size, :]
                patch_mask_binary = make_binary_mask(patch_mask_arr)

                patch_targets = []
                for target in range(7):
                    single_mask = _make_single_target_mask(patch_mask_binary, target)
                    percentage_cover = len(single_mask.nonzero()) / single_mask.numel()
                    patch_targets.append(percentage_cover)
                assert abs(sum(patch_targets) - 1) < 1e-6

                patch_data = [image_id, i_x, i_y] + patch_targets
                all_patch_data.append(patch_data)

    # Save the patch dataframe
    df_cols = ['image_id', 'i_x', 'i_y'] + [DgrLucDataset.target_to_name(i) for i in range(7)]
    patches_df = pd.DataFrame(data=all_patch_data, columns=df_cols)
    patches_df.to_csv(PATCH_DATA_CSV_FMT.format(patch_details.cell_size), index=False)


def _calculate_dataset_normalisation(metadata_df):
    print('Calculating dataset normalisation')
    avgs = []
    for i in tqdm(range(len(metadata_df)), desc='Calculating dataset normalisation', leave=False):
        sat_path = metadata_df['sat_image_path'][i]
        sat_img = Image.open(sat_path)
        sat_img_arr = np.array(sat_img) / 255
        avg = np.mean(sat_img_arr, axis=(0, 1))
        avgs.append(avg)
    arrs = np.stack(avgs)
    arrs_mean = np.mean(arrs, axis=0)
    arrs_std = np.std(arrs, axis=0)
    print(' Mean:', arrs_mean)
    print('  Std:', arrs_std)


def make_binary_mask(mask):
    # TODO technically this is a thresholded mask; binary is a bit misleading.
    binary_mask = torch.zeros_like(torch.as_tensor(mask))
    binary_mask[mask > 128] = 1
    return binary_mask


def _make_single_target_mask(mask_binary, target_clz):
    # mask_img should already be binary
    assert mask_binary.min() >= 0
    assert mask_binary.max() <= 1
    rgb = DgrLucDataset.target_to_rgb(target_clz)
    c1 = mask_binary[:, :, 0] == rgb[0]
    c2 = mask_binary[:, :, 1] == rgb[1]
    c3 = mask_binary[:, :, 2] == rgb[2]
    new_mask = (c1 & c2 & c3)
    return new_mask


def _visualise_data(metadata_df, n_to_show=5):
    random_idxs = np.random.choice(len(metadata_df), size=n_to_show, replace=False)
    for i in random_idxs:
        sat_path = metadata_df['sat_image_path'][i]
        mask_path = metadata_df['mask_path'][i]
        sat_img = Image.open(sat_path)
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        mask_binary = make_binary_mask(mask_arr)

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        axes[0][0].imshow(sat_img, vmin=0, vmax=255)
        axes[0][0].set_title("Satellite Image")
        axes[0][1].imshow(mask_binary.float(), vmin=0, vmax=1)
        axes[0][1].set_title("Complete Mask")
        for target in range(7):
            single_mask = _make_single_target_mask(mask_binary, target)
            axes[(target + 2) // 3][(target + 2) % 3].imshow(single_mask, vmin=0, vmax=1, cmap='gray')
            axes[(target + 2) // 3][(target + 2) % 3].set_title(DgrLucDataset.target_to_name(target))
        plt.tight_layout()
        plt.show()


def _generate_per_class_coverage(metadata_df):
    print('Generating class distribution')
    if os.path.exists(CLASS_DIST_PATH):
        print(' Skipping...')
        return

    cover_dist_df = metadata_df[['image_id']].copy()
    for i in range(7):
        cover_dist_df[DgrLucDataset.target_to_name(i)] = pd.Series(dtype=float)

    for i in tqdm(range(len(metadata_df)), desc='Calculating class coverage for each image', leave=False):
        mask_path = metadata_df['mask_path'][i]
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        mask_binary = make_binary_mask(mask_arr)

        s = 0
        for target in range(7):
            single_mask = _make_single_target_mask(mask_binary, target)
            name = DgrLucDataset.target_to_name(target)
            percentage_cover = len(single_mask.nonzero())/single_mask.numel()
            cover_dist_df.loc[i, name] = percentage_cover
            s += percentage_cover
        assert abs(s - 1) < 1e-6
    cover_dist_df.to_csv(CLASS_DIST_PATH, index=False)


def _plot_per_class_coverage():
    cover_dist_df = DgrLucDataset.load_per_class_coverage()
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(14, 2))
    for target in range(7):
        name = DgrLucDataset.target_to_name(target)
        dist = cover_dist_df[name]
        axes[target].set_title(DgrLucDataset.target_to_name(target))
        axes[target].hist(dist, bins=25, range=(0, 1), log=True)
        axes[target].set_xlabel('Coverage')
        axes[target].set_ylabel('Density')
        axes[target].set_ylim(top=1000)
    plt.tight_layout()
    fig.savefig(CLASS_DIST_PATH.replace("csv", "png"), format='png', dpi=300)
    plt.show()


def _baseline_performance():

    def performance_for_dataset(pred, dataset):
        targets = dataset.targets
        preds = torch.ones_like(targets)
        preds *= pred
        results = RegressionMetric.calculate_metric(preds, targets, None)
        return results

    idx = 0
    results_arr = np.full((1, 5, 3), np.nan, dtype=object)
    for train_dataset, val_dataset, test_dataset in DgrLucDataset.create_datasets():
        train_mean_target = train_dataset.targets.mean(dim=0)
        train_results = performance_for_dataset(train_mean_target, train_dataset)
        val_results = performance_for_dataset(train_mean_target, val_dataset)
        test_results = performance_for_dataset(train_mean_target, test_dataset)
        results_arr[:, idx, :] = [train_results, val_results, test_results]
        idx += 1
    output_regression_results(['Baseline'], results_arr)


class DgrLucDataset(MilDataset, ABC):

    d_in = 1200
    n_expected_dims = 4  # i x c x h x w
    n_classes = 7
    metric_clz = RegressionMetric
    class_dict_df = _load_class_dict_df()
    dataset_mean = (0.4082, 0.3791, 0.2816)
    dataset_std = (0.06722, 0.04668, 0.04768)
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    def __init__(self, bags, targets, instance_targets, bags_metadata, mask_paths):
        super().__init__(bags, targets, instance_targets, bags_metadata)
        self.mask_paths = mask_paths
        self.transform = self.basic_transform

    @classmethod
    @property
    @abstractmethod
    def model_type(cls) -> str:
        pass

    @classmethod
    @property
    @abstractmethod
    def patch_details(cls) -> PatchDetails:
        pass

    @classmethod
    def load_dgr_bags(cls):
        patches_df = pd.read_csv(PATCH_DATA_CSV_FMT.format(cls.patch_details.cell_size))
        coverage_df = cls.load_per_class_coverage()
        metadata_df = _load_metadata_df()

        # Parse instance targets
        instance_targets = []
        for image_id in coverage_df['image_id']:
            image_patch_data = patches_df.loc[patches_df['image_id'] == image_id]
            bag_instance_targets = image_patch_data[cls.get_clz_names()].to_numpy()
            instance_targets.append(bag_instance_targets)

        bags = metadata_df['sat_image_path'].to_numpy()
        bags_metadata = np.asarray([{'id': id_} for id_ in metadata_df['image_id'].tolist()])
        targets = coverage_df[cls.get_clz_names()].to_numpy()
        mask_paths = metadata_df['mask_path'].to_numpy()

        return bags, targets, instance_targets, bags_metadata, mask_paths

    @classmethod
    def target_to_rgb(cls, target):
        r = cls.class_dict_df.loc[cls.class_dict_df['target'] == target]
        rgb = r[['r', 'g', 'b']].values.tolist()[0]
        return rgb

    @classmethod
    def create_clz_palette(cls):
        clz_palette = np.asarray([cls.target_to_rgb(clz) for clz in range(7)]) * 255
        return clz_palette.flatten().tolist()

    @classmethod
    def rgb_to_target(cls, r, g, b):
        row = cls.class_dict_df.loc[(cls.class_dict_df['r'] == r) &
                                    (cls.class_dict_df['g'] == g) &
                                    (cls.class_dict_df['b'] == b)]
        target = row['target'].values.tolist()
        if len(target) != 1:
            raise ValueError("Invalid output for rgb to target: {:}. RGB: {:}".format(target, (r, g, b)))
        return target[0]

    @classmethod
    def target_to_name(cls, target):
        return cls.class_dict_df['name'][target]

    @staticmethod
    def load_per_class_coverage():
        return pd.read_csv(CLASS_DIST_PATH)

    @classmethod
    def create_complete_dataset(cls):
        bags, targets, instance_targets, bags_metadata, mask_paths = cls.load_dgr_bags()
        return cls(bags, targets, instance_targets, bags_metadata, mask_paths)

    @classmethod
    def create_datasets(cls, random_state=12):
        bags, targets, instance_targets, bags_metadata, mask_paths = cls.load_dgr_bags()

        for train_split, val_split, test_split in cls.get_dataset_splits(bags, targets, random_state=random_state):
            # Setup bags, targets, and metadata for splits
            train_bags, val_bags, test_bags = [bags[i] for i in train_split], \
                                              [bags[i] for i in val_split], \
                                              [bags[i] for i in test_split]
            train_targets, val_targets, test_targets = targets[train_split], targets[val_split], targets[test_split]
            train_its, val_its, test_its = [instance_targets[i] for i in train_split], \
                                           [instance_targets[i] for i in val_split], \
                                           [instance_targets[i] for i in test_split]
            train_md, val_md, test_md = bags_metadata[train_split], bags_metadata[val_split], bags_metadata[test_split]
            train_mp, val_mp, test_mp = mask_paths[train_split], mask_paths[val_split], mask_paths[test_split]

            train_dataset = cls(train_bags, train_targets, train_its, train_md, train_mp)
            val_dataset = cls(val_bags, val_targets, val_its, val_md, val_mp)
            test_dataset = cls(test_bags, test_targets, test_its, test_md, test_mp)

            yield train_dataset, val_dataset, test_dataset

    @classmethod
    def get_dataset_splits(cls, bags, targets, random_state=5):
        # Split using stratified k fold (5 splits)
        skf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = skf.split(bags, targets)

        # Split further into train/val/test (80/10/10)
        for train_split, test_split in splits:

            # Split val split (currently 20% of data) into 10% and 10% (so 50/50 ratio)
            val_split, test_split = train_test_split(test_split, random_state=random_state, test_size=0.5)
            # Yield splits
            yield train_split, val_split, test_split

    # TODO should this be a property?
    @classmethod
    def get_clz_names(cls):
        return cls.class_dict_df['name'].tolist()

    @classmethod
    def get_target_mask(cls, instance_targets, clz):
        pass

    @overrides
    def summarise(self, out_clz_dist=True):
        print('- MIL Dataset Summary -')
        print(' {:d} bags'.format(len(self.bags)))

        if out_clz_dist:
            print(' Class Distribution')
            for clz in range(self.n_classes):
                print('  Class {:d} - {:s}'.format(clz, self.get_clz_names()[clz]))
                clz_targets = self.targets[:, clz]
                hist, bins = np.histogram(clz_targets, bins=np.linspace(0, 1, 11))
                for i in range(len(hist)):
                    print('   {:.1f}-{:.1f}: {:d}'.format(bins[i], bins[i + 1], hist[i]))

        bag_sizes = [len(b) for b in self.bags]
        print(' Bag Sizes')
        print('  Min: {:d}'.format(min(bag_sizes)))
        print('  Avg: {:.1f}'.format(np.mean(bag_sizes)))
        print('  Max: {:d}'.format(max(bag_sizes)))

    def get_mask_img(self, bag_idx):
        mask_path = self.mask_paths[bag_idx]
        mask_img = Image.open(mask_path)
        return mask_img

    def get_sat_img(self, bag_idx):
        sat_path = self.bags[bag_idx]
        sat_img = Image.open(sat_path)
        return sat_img

    def __getitem__(self, bag_idx):
        # Load original satellite and mask images
        sat_path = self.bags[bag_idx]

        # Resize to new target resolution
        sat_img = Image.open(sat_path)
        sat_img.thumbnail((self.patch_details.effective_resolution, self.patch_details.effective_resolution))
        sat_img_arr = np.array(sat_img)

        # Iterate through each cell in the grid
        instances = []
        n_x = int(sat_img_arr.shape[0]/self.patch_details.patch_size)
        n_y = int(sat_img_arr.shape[1]/self.patch_details.patch_size)
        for i_x in range(n_x):
            for i_y in range(n_y):
                # Extract patch from original image
                p_x = i_x * self.patch_details.patch_size
                p_y = i_y * self.patch_details.patch_size
                instance = sat_img_arr[p_x:p_x+self.patch_details.patch_size,
                                       p_y:p_y+self.patch_details.patch_size,
                                       :]
                if self.transform is not None:
                    instance = self.transform(instance)
                instances.append(instance)
        instances = torch.stack(instances)
        target = self.targets[bag_idx]
        instance_targets = self.instance_targets[bag_idx]
        mask_path = self.mask_paths[bag_idx]
        return instances, target, instance_targets, mask_path


class DgrLucDataset8Small(DgrLucDataset):
    model_type = "8_small"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset8Medium(DgrLucDataset):
    model_type = "8_medium"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset8Large(DgrLucDataset):
    model_type = "8_large"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset16Small(DgrLucDataset):
    model_type = "16_small"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset16Medium(DgrLucDataset):
    model_type = "16_medium"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset16Large(DgrLucDataset):
    model_type = "16_large"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset24Small(DgrLucDataset):
    model_type = "24_small"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset24Medium(DgrLucDataset):
    model_type = "24_medium"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDataset24Large(DgrLucDataset):
    model_type = "24_large"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDatasetResNet(DgrLucDataset):
    model_type = "resnet"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDatasetUNet224(DgrLucDataset):
    model_type = "unet224"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


class DgrLucDatasetUNet448(DgrLucDataset):
    model_type = "unet448"
    name = "dgr_luc_" + model_type
    patch_details = get_patch_details(model_type)


if __name__ == "__main__":
    setup("8_small")
