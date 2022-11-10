import csv

import latextable
import numpy as np
from texttable import Texttable


def run():
    file = "results/raw_results.txt"
    model_names = ['resnet', 'unet224', 'unet448', '8_small', '8_medium', '8_large', '16_small', '16_medium', '16_large',
                   '24_small', '24_medium', '24_large']
    with open(file, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        split = []
        split_idx = 0
        for row in reader:
            split.append(row)
            if len(split) == 28:
                if split_idx == 0:
                    scene_rmse = parse_split(split, -3)
                    scene_mae = parse_split(split, -2)
                elif split_idx == 1:
                    grid_seg = parse_split(split, -2)
                elif split_idx == 2:
                    high_res_seg = parse_split(split, -2)
                    break
                split_idx += 1
                split = []
                next(reader)

    rows = [['Configuration', 'Scene RMSE', 'Scene MAE', 'Patch mIoU', 'Pixel mIoU']]
    means = []
    for model in model_names:
        row = [format_model_type(model), scene_rmse[model], scene_mae[model], grid_seg[model], high_res_seg[model]]
        row_means = [float(r.split(' +- ')[0]) for r in row[1:]]
        means.append(row_means)
        row[1:] = ['{:.3f} $\pm$ {:.3f}'.format(*[float(s) for s in r.split(' +- ')]) for r in row[1:]]
        row = [r if 'nan' not in r else 'N/A' for r in row]
        rows.append(row)

    for c in range(4):
        col_idx = c + 1
        cell_values = []
        for r in range(len(model_names)):
            row_idx = r + 1
            cell_value = rows[row_idx][col_idx]
            if cell_value == 'N/A':
                cell_values.append(np.nan)
            else:
                cell_value = float(cell_value[:5])
                cell_values.append(cell_value)

        best_val = np.nanmin(cell_values) if col_idx < 3 else np.nanmax(cell_values)
        best_idxs = [r + 1 for r in range(len(model_names)) if cell_values[r] == best_val]
        for best_idx in best_idxs:
            rows[best_idx][col_idx] = '\\textbf{' + rows[best_idx][col_idx] + '}'

    table = Texttable()

    table.set_cols_dtype(['t'] * 5)
    table.set_cols_align(['l'] * 5)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table, use_booktabs=True))

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))
    # print(means)
    # axes[0][0].bar([0, 1, 2, 4, 5, 6], [m[0] for m in means][1:], color=['r', 'g', 'b', 'r', 'g', 'b'])
    # axes[0][0].set_xticks([1, 5])
    # axes[0][0].set_xticklabels(['Grid Size = 16', 'Grid Size = 24'])
    # axes[0][0].set_ylabel('Scene RMSE')
    # axes[0][0].set_ylim(0, 0.12)
    #
    # axes[0][1].bar([0, 1, 2, 4, 5, 6], [m[1] for m in means][1:])
    # axes[0][1].set_xticks([1, 5])
    # axes[0][1].set_xticklabels(['Grid Size = 16', 'Grid Size = 24'])
    # axes[0][1].set_ylabel('Scene MAE')
    # axes[0][1].set_ylim(0, 0.12)
    #
    # axes[1][0].bar([0, 1, 2, 4, 5, 6], [m[2] for m in means][1:])
    # axes[1][0].set_xticks([1, 5])
    # axes[1][0].set_xticklabels(['Grid Size = 16', 'Grid Size = 24'])
    # axes[1][0].set_ylabel('Patch mIoU (Grid)')
    # axes[1][0].set_ylim(0, 0.45)
    #
    # axes[1][1].bar([0, 1, 2, 4, 5, 6], [m[3] for m in means][1:])
    # axes[1][1].set_xticks([1, 5])
    # axes[1][1].set_xticklabels(['Grid Size = 16', 'Grid Size = 24'])
    # axes[1][1].set_ylabel('Patch mIoU (Original)')
    # axes[1][1].set_ylim(0, 0.45)
    #
    # plt.tight_layout()
    # plt.show()


def parse_split(split, idx):
    data = split[4::2]
    model_names = [row[1].strip() for row in data]
    values = [row[idx].strip() for row in data]
    values_dict = dict(zip(model_names, values))
    return values_dict


def format_model_type(model_type):
    if model_type == 'resnet':
        return 'ResNet18'
    elif 'unet' in model_type:
        return 'UNet {:s}'.format(model_type[-3:])
    grid_size, patch_size = model_type.split('_')
    return 'S2P {:s} {:s}'.format(patch_size.title(), grid_size)


if __name__ == "__main__":
    run()
