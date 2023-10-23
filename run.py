import click

from data import MVTecDataset, mvtec_classes, original_one_nut, original_one_screw, metal_nut_13, screw_13, metal_nut_all,screw_all, screw_p, metal_nut_p
from models import PatchCore    
from utils import print_and_export_results

from typing import List

# seeds
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings  
warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
o_m = original_one_nut()
o_s = original_one_screw()
m_13 = metal_nut_13()
s_13 = screw_13()
m_all = metal_nut_all()
s_all = screw_all()
s_p = screw_p()
m_p = metal_nut_p()

def run_model(classes: List, backbone: str):
    results = {}

    for cls in classes:
        print(f"\n█│ Running PatchCore on {cls} dataset.")  
        print(f" ╰{'─' * (len('PatchCore') + len(cls) + 29)}\n")

        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

        model = PatchCore(
            f_coreset=.10,
            backbone_name=backbone,  # Use the backbone specified
        )

        print("   Training ...")
        model.fit(train_ds)  # Train on the training set
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds, dataset_name=cls)

        print(f"\n   ╭{'─' * (len(cls) + 15)}┬{'─' * 20}┬{'─' * 20}╮")
        print(f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
        print(f"   ╰{'─' * (len(cls) + 15)}┴{'─' * 20}┴{'─' * 20}╯")

        results[cls] = [float(image_rocauc), float(pixel_rocauc)]

    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results) / len(image_results)
    pixel_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(pixel_results) / len(pixel_results)
    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results


@click.command()
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
@click.option("--backbone", default="wide_resnet50_2", help="Backbone model, defaults to wide_resnet50_2.")

def cli_interface(dataset: str, backbone: str):
    dataset_dict = {
        'all': ALL_CLASSES,
        'exp_0_nut': o_m,
        'exp_0_screw': o_s,
        'exp_1_nut': m_13,
        'exp_1_screw': s_13, 
        'exp_2_nut': m_all,
        'exp_2_screw': s_all,
        'exp_3_nut': m_p,
        'exp_3_screw': s_p,
    }

    dataset = dataset_dict.get(dataset, [dataset])

    total_results = run_model(dataset, backbone)

    print_and_export_results(total_results, "PatchCore")


if __name__ == "__main__":
    cli_interface()
