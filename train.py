# 필요 library들을 import합니다.
from config import *
from dataset import CustomDataset
from transform import TransformSelector
from model import ModelSelector
from trainer import Loss, Trainer, LabelSmoothingLoss, FocalLoss, FocalCosineLoss

import random
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 재현성 유지
    set_seed(2024)

    # 학습에 사용할 장비를 선택.
    # torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_info = pd.read_csv(traindata_info_file)
    
    # 총 class의 수를 측정.
    num_classes = len(train_info['target'].unique())
    
    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=2024
    )
    
    # 학습에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = transform_type
    )
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)
    
    # 학습에 사용할 Dataset을 선언.
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )
    
    # 학습에 사용할 DataLoader를 선언.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # 학습에 사용할 Model을 선언.
    model_selector = ModelSelector(
        model_type=model_type, 
        num_classes=num_classes,
        model_name=model_name, 
        pretrained=True
    )
    model = model_selector.get_model()
    
    # 선언된 모델을 학습에 사용할 장비로 셋팅.
    model.to(device)
    
    # 학습에 사용할 optimizer를 선언하고, learning rate를 지정
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )
    
    # 스케줄러 초기화
    scheduler_step_size = scheduler_step_size  # 매 30step마다 학습률 감소
    scheduler_gamma = scheduler_gamma  # 학습률을 현재의 10%로 감소
    
    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)
    
    # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
    epochs_per_lr_decay = epochs_per_lr_decay
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )
    
    # 학습에 사용할 Loss를 선언.
    loss_fn = Loss()
    
    # 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=epochs,
        result_path=save_result_path
    )
    
    # 모델 학습.
    trainer.train()

if __name__ == "__train__": # python train.py
    main()