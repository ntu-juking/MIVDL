import os
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import Image, ImageFile
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_filenames = [filename for filename in os.listdir(path)]
        self.label = [filename[-5] for filename in os.listdir(path)]
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.path, img_filename)
        image = Image.open(img_path)
        label = self.image_filenames[idx]
        return image, label

    def _is_image_file(self, filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))

class ImageDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super(ImageDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.custom_collate_fn  # 使用自定义的 collate_fn
        )

    def custom_collate_fn(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return images, labels # 直接返回 batch，保持每个 batch 中的数据格式不变
def Data_loader(path):
    image_dataset = CustomDataset(path)
    data_loader = ImageDataLoader(image_dataset, batch_size = 32)
    return data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../code-slicer/png/')
    parser.add_argument('--batch_size', default='32')
    parser.add_argument('--num_epochs', default='10', type=int)
    parser.add_argument('--lr', default='0.001', type=float, help="The model architecture to be fine-tuned.")
    args = parser.parse_args()
    path = args.data_dir
    data_loader = Data_loader(path)
    length = CustomDataset(path).__len__()
    model = CLIPModel.from_pretrained("../openai/clip-vit-large-patch14/")
    processor = CLIPProcessor.from_pretrained("../openai/clip-vit-large-patch14/")

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    learning_rate = args.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            transform = transforms.ToTensor()
            images = [transform(img) for img in images]
            inputs = [img.to(device) for img in images]
            labels = [lab.to(device) for lab in labels]

            optimizer.zero_grad()
            running_loss = 0.0
            running_corrects = 0

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / length
        epoch_acc = running_corrects.double() / length

        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
