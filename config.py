# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
traindata_dir = "./data/train"
traindata_info_file = "./data/train.csv"
save_result_path = "./train_result"

batch_size = 64
learning_rate = 1e-4
epochs = 15

scheduler_step_size = 30
scheduler_gamma = 0.1
epochs_per_lr_decay = 2
eta_min=1e-6
model_type = 'timm'
model_name='resnet18'