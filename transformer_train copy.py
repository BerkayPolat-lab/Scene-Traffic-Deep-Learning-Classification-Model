# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import sys
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


sys.stdout = open('transformer_output.txt', 'a')  
sys.stderr = open('transformer_error.txt', 'a')
# %%
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# %%
data = pd.read_csv('TrainingSet_37.csv')
test = pd.read_csv('TestSet_37.csv')
# data.head(15)

# %%
# print(test.head(15))
# print(data.shape)
# print(test.shape)

# %%
# print(data['custom_label_path'].nunique())
# print(data.isnull().count().sort_values(ascending=False))
# print(data.columns.tolist())

def scene_label_generator(label_path):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    scene_list = label_data.get('attributes', {}).get('scene', [])
    scene_str = ', '.join(scene_list)
    return scene_str

def traffic_label_generator(label_path):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    scene_list = label_data.get('attributes', {}).get('traffic', [])    
    scene_str = ', '.join(scene_list)
    return scene_str

# %%
def label_engineering(data):
    data['scene_label'] = data.apply(lambda x: scene_label_generator(x['custom_label_path']), axis=1)
    data['traffic_label'] = data.apply(lambda x: traffic_label_generator(x['custom_label_path']), axis=1)
    return data

data = label_engineering(data)
test = label_engineering(test)

# %%
# data.to_csv('./datasets/prepped_data.csv', index=False)
# test.to_csv('./datasets/prepped_test.csv', index=False)

# %%
# print(data.columns.tolist())
# print(data['scene_label'].value_counts())
# print(data['traffic_label'].value_counts())

# %%
scene_classes = ['scene_classes_placeholder']
traffic_classes = ['traffic_classes_placeholder']

# %%
# data = pd.read_csv('./datasets/prepped_data.csv')
# test = pd.read_csv('./datasets/prepped_test.csv')


# %%

IMAGE_SIZE = 224
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiOutputDataset(Dataset):
    def __init__(self, transformer_data, scene_classes, traffic_classes, transform=None):
        self.transformer_data = transformer_data
        self.transform = transform or transforms.ToTensor()
        self.scene_classes = scene_classes
        self.traffic_classes = traffic_classes

        self.scene2idx = {label: idx for idx, label in enumerate(self.scene_classes)}
        self.traffic2idx = {label: idx for idx, label in enumerate(self.traffic_classes)}

    def __len__(self):
        return len(self.transformer_data)
    
    def __getitem__(self, index):
        row = self.transformer_data.iloc[index]
        image_path = row['file_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print('Bad image path', image_path)
            raise RuntimeError(f"Failed to open image: {image_path}") from e

        if self.transform:
            image = self.transform(image)

        try:
            scene_label = torch.tensor(self.scene2idx.get(row['scene_label']), dtype=torch.long)
            traffic_label = torch.tensor(self.traffic2idx.get(row['traffic_label']), dtype=torch.long)
        except KeyError as e:
            print(f"Label error at index {index} - {e}")
            print(row)
            raise

        return image, scene_label, traffic_label
    

weights = ViT_B_16_Weights.IMAGENET1K_V1  
transform = weights.transforms()

dataset = MultiOutputDataset(transformer_data=data, scene_classes=scene_classes, traffic_classes=traffic_classes, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)

testset = MultiOutputDataset(transformer_data=test, scene_classes=scene_classes, traffic_classes=traffic_classes, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=16)

# %%
from torchvision.models import ViT_B_16_Weights

class MultiOutputTransformer(nn.Module):
    def __init__(self, base_model, num_scene, num_traffic):
        super().__init__()
        self.backbone = base_model
        self.backbone.heads = nn.Identity()
        
        in_features = self.backbone.hidden_dim
        self.scene_head = nn.Linear(in_features, num_scene)
        self.traffic_head = nn.Linear(in_features, num_traffic)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.scene_head(x), self.traffic_head(x)

vit_base = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model = MultiOutputTransformer(vit_base, len(scene_classes), len(traffic_classes)).to(DEVICE)

# %%
scene_onehot = pd.get_dummies(data['scene_label'])
traffic_onehot = pd.get_dummies(data['traffic_label'])

scene_onehot = scene_onehot.reindex(columns=scene_classes, fill_value=0)
traffic_onehot = traffic_onehot.reindex(columns=traffic_classes, fill_value=0)

multi_hot_df = pd.concat([scene_onehot, traffic_onehot], axis=1)

train_labels = multi_hot_df.to_numpy() 

num_classes = train_labels.shape[1]
pos_weights = []

for i in range(num_classes):
    class_labels = train_labels[:, i]  
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=class_labels)
    pos_weights.append(weights[1])

pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)
scene_pos_weights = pos_weights_tensor[:len(scene_classes)].to(DEVICE)
traffic_pos_weights = pos_weights_tensor[len(scene_classes):].to(DEVICE)

scene_alpha_tensor = scene_pos_weights / scene_pos_weights.sum()
traffic_alpha_tensor = traffic_pos_weights / traffic_pos_weights.sum()

# %%
print("Model output shape:", model.scene_head)

# %%
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
# %%

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self,  logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()


        BCE_loss = F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction='none')

        pt = torch.where(targets == 1, probs, 1 - probs)  
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
            focal_weight *= alpha_weight

        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# %%
scene_focal_loss = FocalLoss(alpha=scene_alpha_tensor)
traffic_focal_loss = FocalLoss(alpha=traffic_alpha_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, scene_labels, traffic_labels in dataloader:
        images = images.to(DEVICE)
        scene_labels = scene_labels.to(DEVICE)
        traffic_labels = traffic_labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            out_scene, out_traffic = model(images)
            scene_labels = F.one_hot(scene_labels, num_classes=out_scene.shape[1]).float()
            traffic_labels = F.one_hot(traffic_labels, num_classes=out_traffic.shape[1]).float()
            loss_scene = scene_focal_loss(out_scene, scene_labels)
            loss_traffic = traffic_focal_loss(out_traffic, traffic_labels)
            loss = loss_scene + loss_traffic

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

# %%
model.eval()

scene_probs, scene_preds, scene_labels_a = [], [], []
traffic_probs, traffic_preds, traffic_labels_a = [], [], []

with torch.no_grad():
    for images, scene_labels, traffic_labels in testloader:
        images = images.to(DEVICE)
        scene_labels = scene_labels.to(DEVICE)
        traffic_labels = traffic_labels.to(DEVICE)
        scene_out, traffic_out = model(images)

        scene_p = F.softmax(scene_out, dim=1)
        print(scene_p)
        traffic_p = F.softmax(traffic_out, dim=1)
        print(traffic_p)

        scene_probs.append(scene_p.cpu())
        traffic_probs.append(traffic_p.cpu())

        scene_preds.append(scene_p.argmax(dim=1).cpu())
        traffic_preds.append(traffic_p.argmax(dim=1).cpu())

        scene_labels_a.append(scene_labels.cpu())
        traffic_labels_a.append(traffic_labels.cpu())

scene_probs = torch.cat(scene_probs).numpy()
scene_preds = torch.cat(scene_preds).numpy()
scene_labels_a = torch.cat(scene_labels_a).numpy()

traffic_probs = torch.cat(traffic_probs).numpy()
traffic_preds = torch.cat(traffic_preds).numpy()
traffic_labels_a = torch.cat(traffic_labels_a).numpy()


# %%
print(traffic_labels_a)

# %%
from sklearn.metrics import classification_report

def evaluate_head(preds, labels, head_name, class_names=None):
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    classification_report_1=classification_report(labels, preds, target_names=class_names)
    
    print(f"\n[{head_name}]")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 Score:  {f1}")
    print(f"Classification Report: {classification_report_1}")

evaluate_head(scene_preds, scene_labels_a, "Scene", scene_classes)
evaluate_head(traffic_preds, traffic_labels_a, "Traffic", traffic_classes)
