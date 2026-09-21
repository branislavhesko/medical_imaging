import dataclasses
import logging
import os

from omegaconf import OmegaConf
from PIL import Image
from timm import create_model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def build_model(model_name: str = "resnet18", num_classes: int = 2):
    model = create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        in_chans=3,
    )
    return model


def build_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

def export_model(model: torch.nn.Module, path: str) -> None:
    """Save *model* as a ``.pt2`` that loads on any device with any batch size.

    Exporting on CPU keeps the archive free of CUDA tensors (a CUDA export
    cannot be loaded by CPU-only torch at all), and the dynamic batch dim lets
    inference run single images even if training used a bigger batch.
    """
    model = model.eval().cpu()
    example = torch.zeros(2, 3, 224, 224)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
    torch.export.save(torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes), path)


def build_test_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((224, 224)),
    ])


@dataclasses.dataclass
class BedSoresClassifierTrainerConfig:
    path_to_train_data: str = "data/data/"
    model_name: str = "efficientnet_b1"
    num_classes: int = -1
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    classes: list[str] | None = None
    
    def __post_init__(self):
        if self.num_classes == -1:
            self.num_classes = len([d for d in os.listdir(self.path_to_train_data) if os.path.isdir(os.path.join(self.path_to_train_data, d))])


def get_train_dataloader(config: BedSoresClassifierTrainerConfig):
    train_dataset = datasets.ImageFolder(
        root=config.path_to_train_data,
        transform=build_transforms()
    )
    if config.num_classes == -1:
        config.num_classes = len(train_dataset.classes)
    if config.classes is None:
        config.classes = train_dataset.classes
    return DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True,
        persistent_workers=True,
    )

class BedSoresClassifierTrainer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = BedSoresClassifierTrainerConfig()
        self.model = build_model(self.config.model_name, self.config.num_classes)
        self.train_dataloader = get_train_dataloader(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger.info(f"Training model on {self.config.device} for {self.config.num_epochs} epochs, num classes: {self.config.num_classes}")
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
    def train(self):
        self.model.to(self.config.device)
        self.model.train()
        print(f"Training model on {self.config.device} for {self.config.num_epochs} epochs, num classes: {self.config.num_classes}")
        best_accuracy = 0
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.train()
            accuracy = 0
            total = 0
            loss_total = 0
            for images, labels in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                accuracy += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                loss_total += loss.item()
            current_accuracy = accuracy/total
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.logger.info(f"Best accuracy: {best_accuracy}, saving model to model_best.pt2")
                export_model(self.model, "model_best.pt2")
                self.model.to(self.config.device)
                OmegaConf.save(self.config, "model_best.yaml")
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {loss_total/total}, Accuracy: {current_accuracy}")
            self.scheduler.step()

if __name__ == "__main__":
    trainer = BedSoresClassifierTrainer()
    trainer.train()