import optuna
import mlflow
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from data import build_dataloader, build_dataset
from model import ret_model
from config import mlflow
from train import ret_model, ret_optimizer, ret_scheduler, train_fn, eval_fn, log_metrics, log_dataset, load_data
from utils import get_run_id

# Define the objective function for Optuna hyperparameter tuning
def objective(trial):
    experiment_id=get_run_id('hypertune')
    with mlflow.start_run(experiment_id=experiment_id, run_name="Hyperparameter Tuning") as run:
        log_dataset("go_emotions")
        go_emotions = load_data("go_emotions")
        data = go_emotions.data
        batch_size=64
        train, valid, test = data["train"].to_pandas(), data["validation"].to_pandas(), data["test"].to_pandas()

        train_dataset, valid_dataset = build_dataset(40)
        train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlflow.log_param("device", device)

        # Define hyperparameter search space
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
        do_prob = trial.suggest_uniform("do_prob", 0.3, 0.5)
        n_epochs = trial.suggest_int("n_epochs", 2, 4)
        # Update hyperparameters
    #     model = ret_model(n_train_steps, dropout_rate)  # Update your model with the new dropout_rate
    #     optimizer = ret_optimizer(model, learning_rate)  # Update your optimizer with the new learning_rate

        n_train_steps = int(len(train_dataset) / 64 * 10)

        model = ret_model(n_train_steps, do_prob)
        optimizer = ret_optimizer(model)
        scheduler = ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)

        best_val_loss = 100

        # Rest of your training loop
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
                print("Model saved as current val_loss is: ", best_val_loss)
            script_dir = os.path.dirname(os.path.abspath('file'))
            models_dir = os.path.join(script_dir, "models")
            conda_dir = os.path.join(script_dir, "dependencies", "conda.yaml")

            # Log the best model with Optuna hyperparameters
            mlflow.pytorch.log_model(
                pytorch_model=model,
                registered_model_name="hypertune-bert-base",
                artifact_path="artifacts",
            )
    return best_val_loss

# Main training code
def trainer():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30,show_progress_bar=False)  

    best_params = study.best_params
    best_metric = study.best_value

    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric("best_metric", best_metric)


if __name__ == "main":
    trainer()
