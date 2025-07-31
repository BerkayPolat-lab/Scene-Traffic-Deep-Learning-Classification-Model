# Multi-Output & Multi-Label Classification Model - README (Markdown)

## Overview

This repository implements a **multi-output, multi-label image classification model** for road scene and traffic analysis. The project leverages:

* **ResNet50**
* **Vision Transformer (ViT-B/16, 16×16 patches)**
* **OpenAI CLIP** (contrastive fine-tuning)

Each image produces two outputs:

1. **Scene classification**
2. **Traffic classification**

---

## 1. Data Preprocessing

Images are resized to **224×224** and labels are integer-encoded for both tasks.

```python
from torch.utils.data import Dataset
from PIL import Image
import torch

class MultiOutputDataset(Dataset):
    def __init__(self, df, scene_classes, traffic_classes, transform=None):
        self.df = df
        self.transform = transform
        self.scene2idx = {label: idx for idx, label in enumerate(scene_classes)}
        self.traffic2idx = {label: idx for idx, label in enumerate(traffic_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        scene_label = self.scene2idx[row['scene_label']]
        traffic_label = self.traffic2idx[row['traffic_label']]
        return image, torch.tensor(scene_label), torch.tensor(traffic_label)
```

---

## 2. Model Architectures

### 2.1 ResNet50 Multi-Output Model

```python
import torch
import torch.nn as nn
from torchvision import models

class MultiOutputResNet(nn.Module):
    def __init__(self, num_scene, num_traffic):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.scene_head = nn.Linear(in_features, num_scene)
        self.traffic_head = nn.Linear(in_features, num_traffic)

    def forward(self, x):
        features = self.base_model(x)
        return self.scene_head(features), self.traffic_head(features)
```

### 2.2 Vision Transformer (ViT-B/16)

```python
from torchvision import models

class MultiOutputTransformer(nn.Module):
    def __init__(self, num_scene, num_traffic):
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        in_features = self.backbone.hidden_dim
        self.scene_head = nn.Linear(in_features, num_scene)
        self.traffic_head = nn.Linear(in_features, num_traffic)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.scene_head(x), self.traffic_head(x)
```

### 2.3 OpenAI CLIP with Contrastive Fine-Tuning

```python
from transformers import CLIPProcessor, CLIPModel, Trainer, TrainingArguments
import torch.nn as nn

class CLIPContrastiveTrainerModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, pixel_values, return_loss=False):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=return_loss
        )
```

Prompts:

```python
data['scene_prompt'] = data['scene_label'].apply(lambda x: f"a photo of a road scene with {x.replace('_', ' ')}")
data['traffic_prompt'] = data['traffic_label'].apply(lambda x: f"a photo of a road with {x.replace('_', ' ')}")
```
---

## 3. Training Logic

### 3.1 PyTorch Training Loop (ResNet / ViT)

```python
import torch.optim as optim

model = MultiOutputResNet(len(scene_classes), len(traffic_classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for images, scene_labels, traffic_labels in train_loader:
        images, scene_labels, traffic_labels = images.to(DEVICE), scene_labels.to(DEVICE), traffic_labels.to(DEVICE)
        optimizer.zero_grad()
        scene_preds, traffic_preds = model(images)
        loss = criterion(scene_preds, scene_labels) + criterion(traffic_preds, traffic_labels)
        loss.backward()
        optimizer.step()
```

### 3.2 CLIP Training with Hugging Face Trainer

```python
training_args = TrainingArguments(
    output_dir="./models/clip_contrastive_finetune",
    per_device_train_batch_size=32,
    num_train_epochs=10,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=CLIPContrastiveTrainerModel(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor
)

trainer.train()
```

---

## 4. Inference

```python
def predict(model, image, transform):
    model.eval()
    with torch.no_grad():
        img = transform(image).unsqueeze(0).to(DEVICE)
        scene_logits, traffic_logits = model(img)
        scene_pred = torch.argmax(scene_logits, dim=1).item()
        traffic_pred = torch.argmax(traffic_logits, dim=1).item()
    return scene_pred, traffic_pred
```

---

## 5. Key Features

* Multi-output classification with shared backbone
* Supports **ResNet50**, **ViT-B/16**, and **OpenAI CLIP**
* Implements **Focal Loss** for imbalanced classes
* Mixed precision training for speed and memory efficiency
* Hugging Face Trainer integration for CLIP fine-tuning

---

## 6. References

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [PyTorch ResNet & ViT](https://pytorch.org/vision/stable/models.html)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers)
