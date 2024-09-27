# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
traindata_dir = "/data/ephemeral/home/data/train"
traindata_info_file = "/data/ephemeral/home/data/train.csv"
save_result_path = "/data/ephemeral/home/data/result"

# 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
testdata_dir = "/data/ephemeral/home/data/test"
testdata_info_file = "/data/ephemeral/home/data/test.csv"
save_result_path = "/data/ephemeral/home/data/result"
save_ensemble_path = "/data/ephemeral/home/data/ensemble"

batch_size = 64
learning_rate = 1e-4
epochs = 15

scheduler_step_size = 30
scheduler_gamma = 0.1
epochs_per_lr_decay = 2
eta_min=1e-6
transform_type = 'albumentations'
model_type = 'timm'
model_name='resnet18'
num_classes = 500