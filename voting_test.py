from config import *
from dataset import CustomDataset
from transform import TransformSelector
from model import ModelSelector

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

def soft_voting_ensemble(
    models: list,  
    device: torch.device,
    dataloaders: list  
):
    models = [model.to(device).eval() for model in models]

    all_predictions = []
    with torch.no_grad():
        for batch_list in tqdm(zip(*dataloaders), total=min([len(loader) for loader in dataloaders]), desc="Processing batches"):
            logits_list = []
            for model, batch in zip(models, batch_list):
                batch = batch.to(device)
                logits = model(batch)
                logits = F.softmax(logits, dim=1)
                logits_list.append(logits)

            avg_logits = torch.mean(torch.stack(logits_list), dim=0)
            preds = avg_logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())

    return all_predictions

def hard_voting_ensemble(predictions_list: list):
    predictions_array = np.array(predictions_list)
    final_predictions = []

    for i in range(predictions_array.shape[1]):
        unique, counts = np.unique(predictions_array[:, i], return_counts=True)
        final_predictions.append(unique[np.argmax(counts)])

    return final_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_info = pd.read_csv(testdata_info_file)

    num_classes = 500
    model_configs = [
        {"model_name": "eva_large_patch14_196.in22k_ft_in22k_in1k", "input_size": 196},
        {"model_name": "convnext_xxlarge.clip_laion2b_soup_ft_in1k", "input_size": 256},
        {"model_name": "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "input_size": 224},
        {"model_name": "resnext101_32x32d", "input_size": 224},
        {"model_name": "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384", "input_size": 384},
        {"model_name": "swin_large_patch4_window7_224", "input_size": 224}
    ]

    transform_selector = TransformSelector(transform_type="albumentations")
    model_loaders = [] 
    models = []

    for config in model_configs:
        model_name = config['model_name']
        input_size = config['input_size']

        test_transform = transform_selector.get_transform(size=input_size, is_train=False)

        test_dataset = CustomDataset(
            root_dir=testdata_dir,
            info_df=test_info,
            transform=test_transform,
            is_inference=True
        )

        dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            drop_last=False
        )

        model_loaders.append(dataloader) 

        model_selector = ModelSelector(
            model_type='timm',
            num_classes=num_classes,
            model_name=model_name,
            pretrained=False
        )
        model = model_selector.get_model()

        model.load_state_dict(
            torch.load(
                os.path.join(save_result_path, f"{model_name}.pt"),
                map_location=device
            )
        )

        models.append(model)

    group1 = [models[4], models[1], models[0], models[2], models[3]]
    group2 = [models[4], models[1], models[0], models[3], models[5]]
    group3 = [models[4], models[1], models[0], models[2], models[5]]

    model_groups = [group1, group2, group3]
    loader_groups = [
        [model_loaders[4], model_loaders[1], model_loaders[0], model_loaders[2], model_loaders[3]],
        [model_loaders[4], model_loaders[1], model_loaders[0], model_loaders[3], model_loaders[5]],
        [model_loaders[4], model_loaders[1], model_loaders[0], model_loaders[2], model_loaders[5]]
    ]

    all_group_predictions = []
    for group, loaders in zip(model_groups, loader_groups):
        group_predictions = soft_voting_ensemble(
            models=group,
            device=device,
            dataloaders=loaders 
        )
        all_group_predictions.append(group_predictions)

    final_predictions = hard_voting_ensemble(all_group_predictions)

    test_info['target'] = final_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(os.path.join(save_result_path, "final_predictions.csv"), index=False)

if __name__ == "__voting_test__": # python voting_test.py
    main()