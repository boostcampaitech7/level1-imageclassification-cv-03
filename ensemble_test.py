from config import *
from dataset import CustomDataset
from transform import TransformSelector
from model import ModelSelector

import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 앙상블 추론을 위한 함수
def inference_ensemble(
    model_loader_pairs: list, 
    device: torch.device, 
):
    sum_of_probs = None
    
    # 모델을 평가 모드로 설정
    for model, loader in model_loader_pairs:
        model.to(device)
        model.eval()
        model_probs = []
        
        with torch.no_grad():  # Gradient 계산을 비활성화
            for images in tqdm(test_loader):
                # 데이터를 같은 장치로 이동
                images = images.to(device)
                
                logits = model(images)
                logits = F.softmax(logits, dim=1)  # 각 모델의 확률 값으로 변환
                model_probs.append(logits)
            
        # 현재 모델의 모든 배치 예측을 합칩니다
        model_probs = torch.cat(model_probs, dim=0)
            
        # 확률을 합산
        if sum_of_probs is None:
            sum_of_probs = model_probs
        else:
            sum_of_probs += model_probs
                
    # 예측 결과 저장
    mean_probs = sum_of_probs / len(model_loader_pairs)
    predictions = torch.argmax(mean_probs, dim=1).cpu().detach().numpy() # 결과를 CPU로 옮기고 리스트에 추가
        
    return predictions

def main():
    # 3개의 최상위 모델을 불러오기.
    model_paths = [
        #os.path.join(save_ensemble_path, "convnext_xxlarge_full.pt"),
        os.path.join(save_ensemble_path, "convnext_large_full.pt"),
        os.path.join(save_ensemble_path, "resnext16.pt"),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_and_loaders = []
    for path in model_paths:
        if 'resnext32' in path:
            mn = 'resnext101_32x32d'
        elif 'resnext16' in path:
            mn = 'resnext101_32x16d'
        elif 'eva' in path:
            mn = 'eva_large_patch14_336'
        elif 'swin' in path:
            mn = 'swin_large_patch4_window7_224'
        elif 'beitv2' in path:
            mn = 'beitv2_large_patch16_224'
        elif 'xxlarge' in path:
            mn = 'convnext_xxlarge'
        elif 'xlarge' in path:
            mn = 'convnext_xlarge'
        elif 'large' in path:
            mn ='convnext_large'
        elif 'beitv2' in path:
            mn = 'beitv2_large_patch16_224'
            
        # 추론에 사용할 Transform을 선언.
        transform_selector = TransformSelector(
            transform_type = "albumentations2",
            model_name=mn
        )
        test_transform = transform_selector.get_transform(is_train=False)
        
        # 추론에 사용할 Dataset을 선언.
        test_dataset = CustomDataset(
            root_dir=testdata_dir,
            info_df=test_info,
            transform=test_transform,
            is_inference=True
        )
    
        # 추론에 사용할 DataLoader를 선언.
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False,
            drop_last=False
        )
        
        model_selector = ModelSelector(
            model_type='timm', 
            num_classes=num_classes,
            model_name=mn,
            pretrained=False
        )
        
        model = model_selector.get_model()
        model.load_state_dict(torch.load(path, map_location=device))
        models_and_loaders.append((model, test_loader))
    
    # predictions를 CSV에 저장할 때 형식을 맞춰서 저장
    # 테스트 함수 호출
    predictions = inference_ensemble(
        model_loader_pairs=models_and_loaders,
        device=device,
    )
    
    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info
    
    # DataFrame 저장
    test_info.to_csv("output.csv", index=False)


if __name__ == "__ensemble_test__": # python ensemble_test.py
    main()