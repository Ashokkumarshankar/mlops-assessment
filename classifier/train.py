from sklearn import metrics, model_selection, preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from preprocess import load_data
from config import mlflow
from data import build_dataloader, build_dataset
from model import ret_model
import os

def ret_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = AdamW(optimizer_parameters, lr=5e-5)
    return opt

def ret_scheduler(optimizer, num_train_steps):
    sch = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    return sch


def loss_fn(outputs, labels):
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss()(outputs, labels.float())

def log_metrics(preds, labels):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    
    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    return {"auc_micro": auc_micro}

def train_fn(data_loader, model, optimizer, device, scheduler):
    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
    return train_loss
    

def eval_fn(data_loader, model, device):
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets

def log_dataset(dataset_name):
    dataset = load_data(dataset_name)
    serialized_dataset = dataset.save_to_disk("dataset_directory")
    mlflow.log_artifact("dataset_directory")



def trainer():
    log_dataset("go_emotions")
    go_emotions = load_data("go_emotions")
    data = go_emotions.data
    train, valid, test = data["train"].to_pandas(), data["validation"].to_pandas(), data["test"].to_pandas()
    train_dataset, valid_dataset = build_dataset(40)
    train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, 64)
    mlflow.log_param("train_dataset_size", len(train))
    mlflow.log_param("valid_dataset_size", len(valid))
    mlflow.log_param("test_dataset_size", len(test))

    mlflow.log_param("dataset_name", "Emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlflow.log_param("device", device)
    n_train_steps = int(len(train_dataset) / 64 * 10)

    model = ret_model(n_train_steps, 0.3)
    optimizer = ret_optimizer(model)
    scheduler = ret_scheduler(optimizer, n_train_steps)
    model.to(device)
    model = nn.DataParallel(model)
    n_epochs = 2

    best_val_loss = 100

    for epoch in tqdm(range(n_epochs)):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)

        auc_score = log_metrics(preds, labels)["auc_micro"]
        avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)
        mlflow.log_metric("AUC", auc_score)
        mlflow.log_metric("Avg Train Loss", avg_train_loss)
        mlflow.log_metric("Avg Valid Loss", avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./best_model.pt")  
            print("Model saved has current val_loss is: ", best_val_loss)
        mlflow.log_artifact(model)
        mlflow.pytorch.log_model(
            pytorch_model= model,
            registered_model_name="best-base",
    )