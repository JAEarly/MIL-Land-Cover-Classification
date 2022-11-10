import os
import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from dgr_luc_dataset import RECONSTRUCTION_DATA_DIR_FMT

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class MilLucInterpretabilityStudy:

    def __init__(self, device, dataset, model, show_outputs):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.show_outputs = show_outputs

    def create_reconstructions(self):
        reconstruction_dir = RECONSTRUCTION_DATA_DIR_FMT.format(self.dataset.cell_size, self.dataset.patch_size)
        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        for idx in tqdm(range(len(self.dataset)), total=len(self.dataset), desc='Creating reconstructions'):
            reconstruction = self.dataset.create_reconstructed_image(idx, add_grid=True)
            file_name = "reconstruction_{:d}_{:d}_{:}.png".format(self.dataset.cell_size, self.dataset.patch_size,
                                                                  self.dataset.bags_metadata[idx]['id'])
            reconstruction.save(reconstruction_dir + "/" + file_name)

    def sample_interpretations(self):
        random_idxs = list(range(len(self.dataset)))
        random.shuffle(random_idxs)

        for idx in tqdm(random_idxs, 'Generating interpretations'):
            data = self.dataset[idx]
            bmd = self.dataset.bags_metadata[idx]
            bag, target = data[0], data[1]
            self.create_interpretation(idx, bag, target, bmd['id'])

    def create_interpretation_from_id(self, img_id):
        print('Looking for image id: {:}'.format(img_id))
        for idx, bag_md in enumerate(self.dataset.bags_metadata):
            if bag_md['id'] == img_id:
                print(' Found')
                data = self.dataset[idx]
                bmd = self.dataset.bags_metadata[idx]
                bag, target = data[0], data[1]
                return self.create_interpretation(idx, bag, target, bmd['id'])
            else:
                continue
        print(' Not found')

    def create_interpretation(self, idx, bag, target, bag_id):
        save_path = "out/interpretability/{:}/{:}_interpretation.png".format(self.dataset.model_type, bag_id)

        bag_prediction, patch_preds = self.model.forward_verbose(bag)
        patch_preds = patch_preds.detach().cpu()
        # print('  Pred:', ['{:.3f}'.format(p) for p in bag_prediction])
        # print('Target:', ['{:.3f}'.format(t) for t in target])

        # Work out max absolute prediction for each class, and create a normaliser for each class
        max_abs_preds = [max(abs(torch.min(patch_preds[c]).item()),
                             abs(torch.max(patch_preds[c]).item()))
                         for c in range(7)]
        norms = [plt.Normalize(-m, m) for m in max_abs_preds]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["red", "lightgrey", self.dataset.target_to_rgb(clz)])
                 for clz in range(7)]

        sat_img = self.dataset.get_sat_img(idx)
        mask_img = self.dataset.get_mask_img(idx)
        _, grid_ground_truth_coloured_mask = self._create_ground_truth_overall_mask(mask_img)
        pred_masks = self._create_overall_mask(sat_img, patch_preds, cmaps, norms)
        pred_clz_mask, overall_colour_mask, overall_weighted_colour_mask, _ = pred_masks

        # Can skip plotting if we're not showing the output and the file already exists
        if self.show_outputs or not os.path.exists(save_path):
            clz_counts = Counter(pred_clz_mask.flatten().tolist()).most_common()
            clz_order = [int(c[0]) for c in clz_counts]

            def format_axis(ax, title=None):
                ax.set_axis_off()
                ax.set_aspect('equal')
                if title is not None:
                    ax.set_title(title, fontsize=16)

            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5))
            axes[0][0].imshow(sat_img)
            format_axis(axes[0][0], "Original Image")

            axes[0][1].imshow(mask_img)
            format_axis(axes[0][1], "True Pixel Mask")

            # If grid size is > 1, then it's worth looking at the patch grid mask
            #  Otherwise, when grid size = 1 (e.g., with UNet), output the unweighted predicted mask
            if self.dataset.patch_details.grid_size > 1:
                axes[0][2].imshow(grid_ground_truth_coloured_mask)
                format_axis(axes[0][2], "True Patch Mask")

                axes[0][3].imshow(overall_weighted_colour_mask)
                format_axis(axes[0][3], "Predicted Mask")
            else:
                axes[0][2].imshow(overall_weighted_colour_mask)
                format_axis(axes[0][2], "Predicted Mask")

                axes[0][3].imshow(overall_colour_mask)
                format_axis(axes[0][3], "Predicted Mask (Unweighted)")

            for clz_idx in range(4):
                axis = axes[1][clz_idx]
                if clz_idx < len(clz_order):
                    clz = clz_order[clz_idx]
                    im = axis.imshow(patch_preds[clz], cmap=cmaps[clz], norm=norms[clz])
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                                        ticks=[-max_abs_preds[clz], 0, max_abs_preds[clz]])
                    cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
                    cbar.ax.tick_params(labelsize=14)
                    format_axis(axis, self.dataset.target_to_name(clz).replace('_', ' ').title())
                else:
                    format_axis(axis, '')

            plt.tight_layout()
            if self.show_outputs:
                plt.show()
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

        return sat_img, mask_img, grid_ground_truth_coloured_mask, overall_colour_mask, overall_weighted_colour_mask

    def _create_overall_mask(self, sat_img, patch_preds, cmaps, norms):
        # Create mask of top clz and its weight for each pixel
        _, grid_size, _ = patch_preds.shape
        overall_weight_mask, overall_clz_mask = torch.max(patch_preds, dim=0)

        # Create palette to map from clz to colours
        clz_palette = self.dataset.create_clz_palette()

        # Create color mask from palette and clz mask
        overall_colour_mask = Image.new('P', (grid_size, grid_size))
        overall_colour_mask.putdata(torch.flatten(overall_clz_mask).numpy())
        overall_colour_mask.putpalette(clz_palette, rawmode="RGB")
        overall_colour_mask = overall_colour_mask.convert('RGB')

        # Create weight palette
        #  Each class is mapped to a different range
        #    Clz 0 -> 0 to 35
        #    Clz 1 -> 36 to 71
        #    etc.
        #  Each class range is then mapped to it's colour map
        #  Max length of a palette is 768 (256 RGB colours), so with 7 classes, the max range is 36 (36 * 7 * 3 = 756)
        weight_palette = []
        for clz in range(7):
            cmap = cmaps[clz]
            for i in range(36):
                val = cmap(i / 36)
                color = [int(c * 255) for c in val[:3]]
                weight_palette.extend(color)

        # Normalise weight mask
        #  Do for all classes first, then chose the correct normed value based on the selected class for each pixel
        norm_overall_weight_masks = [norm(overall_weight_mask) for norm in norms]
        norm_overall_weight_mask = np.zeros_like(overall_weight_mask)
        for clz in range(7):
            norm_overall_weight_mask = np.where(overall_clz_mask == clz,
                                                norm_overall_weight_masks[clz],
                                                norm_overall_weight_mask)

        # Convert the weight mask to match the weight palette values
        #  Convert to range (0 to 35) and round to nearest int
        rounded_norm_overall_weight_mask = np.floor(norm_overall_weight_mask * 35).astype(int)
        #  Add clz mask values (multiplied by 36 to map to range start values)
        overall_weight_mask_p = rounded_norm_overall_weight_mask + (overall_clz_mask * 36).numpy()

        # Create weighted color mask from palette and clz mask
        overall_weighted_colour_mask = Image.new('P', (grid_size, grid_size))
        overall_weighted_colour_mask.putdata(overall_weight_mask_p.flatten())
        overall_weighted_colour_mask.putpalette(weight_palette, rawmode="RGB")
        overall_weighted_colour_mask = overall_weighted_colour_mask.convert('RGB')

        overlay = overall_colour_mask.resize(sat_img.size, Image.NEAREST)
        overlay.putalpha(int(0.5 * 255))
        sat_img_with_overlay = sat_img.convert('RGBA')
        sat_img_with_overlay.paste(overlay, (0, 0), overlay)

        return overall_clz_mask, overall_colour_mask, overall_weighted_colour_mask, sat_img_with_overlay

    def _create_ground_truth_overall_mask(self, original_mask):
        grid_size = self.dataset.patch_details.grid_size
        cell_size = self.dataset.patch_details.cell_size
        sat_img_arr = np.array(original_mask)

        overall_clz_mask = np.zeros((grid_size, grid_size))
        overall_coloured_mask = np.zeros((grid_size, grid_size, 3))
        for i_x in range(grid_size):
            for i_y in range(grid_size):
                # Extract patch from original image
                p_x = i_x * cell_size
                p_y = i_y * cell_size
                patch_img_arr = sat_img_arr[p_x:p_x + cell_size, p_y:p_y + cell_size, :]

                # Get max colour in this patch and work out which class it is
                colours, counts = np.unique(patch_img_arr.reshape(-1, 3), axis=0, return_counts=1)
                top_idx = np.argmax(counts)
                top_colour = colours[top_idx]
                clz = self.dataset.rgb_to_target(top_colour[0] / 255, top_colour[1] / 255, top_colour[2] / 255)

                # Update masks
                overall_clz_mask[i_x, i_y] = clz
                overall_coloured_mask[i_x, i_y, 0] = top_colour[0] / 255
                overall_coloured_mask[i_x, i_y, 1] = top_colour[1] / 255
                overall_coloured_mask[i_x, i_y, 2] = top_colour[2] / 255

        return overall_clz_mask, overall_coloured_mask
