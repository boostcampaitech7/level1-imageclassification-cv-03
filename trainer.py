import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss,
        epochs: int,
        result_path: str,
        patience: int = 7,  # Early Stopping을 위한 patience (10번 이상 개선되지 않으면 중단)
        min_delta = 0.0001
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []  # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf')  # 가장 낮은 Loss를 저장할 변수
        self.min_delta = min_delta

        # Early Stopping 관련 변수
        self.patience = patience  # 성능 개선이 없는 에폭 수
        self.counter = 0  # 개선되지 않은 에폭 카운터
        self.early_stop = False  # Early Stopping 여부

    def save_model(self, epoch, loss):      
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if self.val_loader is None:
            model_path = os.path.join(self.result_path, f'All_train_model{epoch}.pt')
            torch.save(self.model.state_dict(), model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")
        else:
            if loss < self.lowest_loss:
                best_model_path = os.path.join(self.result_path, 'best_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()

        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def check_early_stopping(self, val_loss):
        if val_loss < self.lowest_loss - self.min_delta:
            self.counter = 0  # 개선되었으므로 counter 초기화
            self.lowest_loss = val_loss # 최저 손실 업데이트
            print('Reset counter')
        else:
            self.counter += 1  # 개선되지 않으면 counter 증가
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True  # Early Stopping 플래그 설정

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            if self.early_stop:
                print("Early stopping, stopping training")
                break
            
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")
                self.save_model(epoch, val_loss)
                
                # Early Stopping 확인
                self.check_early_stopping(val_loss)
            else:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f} (No validation)\n")
                self.save_model(epoch, train_loss)
                
            # 학습률 스케줄러 업데이트
            self.scheduler.step()
            print(f'lowest_loss: {self.lowest_loss}\n')
