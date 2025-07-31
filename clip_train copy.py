# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import sys
os.environ["USE_TF"] = "0"

sys.stdout = open('./output/clip_output.txt', 'a')  
sys.stderr = open('./output/clip_error.txt', 'a')
# %%
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# %%
data = pd.read_csv('TrainingSet_37.csv')
test = pd.read_csv('TestSet_37.csv')

data.head()

# %%
print(data.loc[:, ['file_path']].sample(n=10, random_state=42))

# %%
data['camera'] = data['file_path'].str.extract(r'/(cam_front_[^/]+)/')
print(data['camera'].value_counts())

# %%
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

def label_engineering(data):
    data['scene_label'] = data.apply(lambda x: scene_label_generator(x['custom_label_path']), axis=1)
    data['traffic_label'] = data.apply(lambda x: traffic_label_generator(x['custom_label_path']), axis=1)
    return data

data = label_engineering(data)
test = label_engineering(test)

# %%
# print(torch.version.cuda)
# print(torch.__version__)

# %%
data = pd.concat([data, test], axis=0)

# %%
import sys
# print(sys.executable)

# %%
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/16', device=device)

# print(model)
# print(preprocess)

# %%
from datasets import Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer

scene_classes = ['scene_classes_placeholder']
traffic_classes = ['traffic_classes_placeholder']
data['scene_prompt'] = data['scene_label'].apply(lambda x: f"a photo of a road scene with {x.replace('_', ' ')}")
data['traffic_prompt'] = data['traffic_label'].apply(lambda x: f"a photo of a road with {x.replace('_', ' ')}")
test['scene_prompt'] = test['scene_label'].apply(lambda x: f"a photo of a road scene with {x.replace('_', ' ')}")
test['traffic_prompt'] = test['traffic_label'].apply(lambda x: f"a photo of a road with {x.replace('_', ' ')}")

scene2idx = {label: idx for idx, label in enumerate(scene_classes)}
traffic2idx = {label: idx for idx, label in enumerate(traffic_classes)}

data['scene_label_num'] = data['scene_label'].map(scene2idx)
data['traffic_label_num'] = data['traffic_label'].map(traffic2idx)
test['scene_label_num'] = test['scene_label'].map(scene2idx)
test['traffic_label_num'] = test['traffic_label'].map(traffic2idx)

#%%
flattened_data = pd.DataFrame({
    "file_path": data["file_path"].tolist() * 2,
    "prompt": data["scene_prompt"].tolist() + data["traffic_prompt"].tolist(),
    "label": data["scene_label_num"].tolist() + data["traffic_label_num"].tolist(), 
    "task": ["scene"] * len(data) + ["traffic"] * len(data)  
})

flattened_test = pd.DataFrame({
    "file_path": test["file_path"].tolist() * 2,
    "prompt": test["scene_prompt"].tolist() + test["traffic_prompt"].tolist(),
    "label": test["scene_label_num"].tolist() + test["traffic_label_num"].tolist(),
    "task": ["scene"] * len(test) + ["traffic"] * len(test)
})

data = Dataset.from_pandas(flattened_data)
test = Dataset.from_pandas(flattened_test)

split_data = data.train_test_split(train_size=0.8, seed=42, shuffle=True)

train_dataset = split_data["train"]
validation_dataset = split_data["test"]

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess(row):
    image = Image.open(row["file_path"]).convert("RGB")
    processed = processor(text=row["prompt"], images=image, return_tensors="pt", padding=True,  truncation=True)
    return {
        "pixel_values": processed["pixel_values"][0],
        "input_ids": processed["input_ids"][0],
        "attention_mask": processed["attention_mask"][0],
        "labels": row["label"]
    }
train_processed = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_processed = validation_dataset.map(preprocess, remove_columns=validation_dataset.column_names)
test_processed = test.map(preprocess, remove_columns=test.column_names)

# Format datasets for PyTorch
train_processed.set_format(type="torch")
val_processed.set_format(type="torch")
test_processed.set_format(type="torch")

# %%
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

class CLIPContrastiveTrainerModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()  
        self.model = CLIPModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, pixel_values, return_loss=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=return_loss)
        return outputs
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./models/clip_contrastive_finetune",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=0.00001,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    remove_unused_columns=False,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True
)

model = CLIPContrastiveTrainerModel()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_processed,
    eval_dataset=val_processed,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()
predictions = trainer.predict(test_processed)
logits, labels, _ = predictions
preds = np.argmax(logits, axis=1)

# %%
from sklearn.metrics import classification_report

df_test = test.to_pandas()

# Identify which rows are scene vs traffic
scene_mask = df_test["task"] == "scene"
traffic_mask = df_test["task"] == "traffic"

print("Scene Classification Report:\n", classification_report(df_test.loc[scene_mask, "label"], preds[scene_mask]))

print("Traffic Classification Report:\n", classification_report(df_test.loc[traffic_mask, "label"], preds[traffic_mask]))
