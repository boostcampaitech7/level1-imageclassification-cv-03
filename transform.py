import torch
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models, datasets, transforms
import torchvision.transforms.v2 as transforms2

class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    transforms.RandomRotation(15),  # 최대 15도 회전
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환
        
        transformed = self.transform(image)  # 설정된 변환을 적용
        
        return transformed  # 변환된 이미지 반환

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

# 여러 실험을 위한 Transform
class AlbumentationsTransform2:
    def __init__(self, model_name: str, is_train: bool = True):
        # 앙상블을 위해 각 모델 configuration 불러오기
        model = timm.create_model(model_name, pretrained=True, num_classes=500)
        model_cfg = model.default_cfg
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(model_cfg['input_size'][1], model_cfg['input_size'][2]),
            A.Normalize(mean=model_cfg['mean'], std=model_cfg['std']),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.4),
                    # A.VerticalFlip(p=0.3),
                    # A.RandomRotate90(p=0.5),
                    A.Rotate(limit=15, p=0.7),
                    
                    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                    # A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
                    
                    # A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
                    # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    # A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
                    
                    # A.GaussNoise(blur_limit=(3, 5), p=0.15),
                    # A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    
                    # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=255, p=0.5),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

# TTA를 위한 Transfrom1
class TTAlbumTrans1:
    def __init__(self, model_name: str, is_train: bool = False):
        model = timm.create_model(model_name, pretrained=True, num_classes=500)
        model_cfg = model.default_cfg
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        self.transform = [
            A.Resize(model_cfg['input_size'][1], model_cfg['input_size'][2]),
            A.Normalize(mean=model_cfg['mean'], std=model_cfg['std']),  # 정규화
            A.HorizontalFlip(p=1),
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

# TTA를 위한 Transfrom1
class TTAlbumTrans2:
    def __init__(self, model_name: str, is_train: bool = False):
        model = timm.create_model(model_name, pretrained=True, num_classes=500)
        model_cfg = model.default_cfg
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        self.transform = [
            A.Resize(model_cfg['input_size'][1], model_cfg['input_size'][2]),
            A.Normalize(mean=model_cfg['mean'], std=model_cfg['std']),  # 정규화
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

# TTA를 위한 Transfrom1
class TTAlbumTrans3:
    def __init__(self, model_name: str, is_train: bool = False):
        model = timm.create_model(model_name, pretrained=True, num_classes=500)
        model_cfg = model.default_cfg
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        self.transform = [
            A.Resize(model_cfg['input_size'][1], model_cfg['input_size'][2]),
            A.Normalize(mean=model_cfg['mean'], std=model_cfg['std']),  # 정규화
            A.Rotate(limit=20, p=1),
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str, model_name: str=None):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations", "albumentations2", "tta1", "tta2", "tta3"]:
            self.transform_type = transform_type
            self.model_name = model_name
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)
        
        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)
        
        elif self.transform_type == 'albumentations2':
            transform = AlbumentationsTransform2(model_name=self.model_name, is_train=is_train)
        
        elif self.transform_type == 'tta1':
            transform = TTAlbumTrans1(model_name=self.model_name, is_train=is_train)
        
        elif self.transform_type == 'tta2':
            transform = TTAlbumTrans2(model_name=self.model_name, is_train=is_train)
        
        elif self.transform_type == 'tta3':
            transform = TTAlbumTrans3(model_name=self.model_name, is_train=is_train)
        
        return transform