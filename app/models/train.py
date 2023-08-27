import os
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
import copy
import time
import json
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import numpy as np
import random

from app.models.model import GCN, AttentionGCN
from app.models.dataset import MyGCNDataset
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, settings_path="./resources/settings.json"):
        super().__init__()

        with open(settings_path, 'r') as f:
            self.settings = json.load(f)

        self.data_path = self.settings.get("data_path", "./resources/data/data.json")
        self.save_dir = self.settings.get("save_dir", "./resources/epochs")
        self.num_epochs = self.settings.get("num_epochs", 20)
        self.seed = self.settings.get("seed", 2022)

    def run(self):
        self.main_train(self.data_path, self.num_epochs, self.save_dir, self.seed)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.enabled = False
        #torch.backends.cudnn.benchmark = False

    def train_model(self, model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs=25, save_dir="checkpoints"):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        summary_writer = SummaryWriter(log_dir="runs")

        for epoch in range(num_epochs):

            # train
            scheduler.step()

            model.train()  # Set model to training mode

            metrics = defaultdict(float)
            metrics['loss'] = 0
            metrics['num_correct'] = 0
            metrics['num_total'] = 0
            epoch_samples = 0
            label_list = []
            predict_list = []

            for data, A, label in train_dataloader:
                data = data.to(device)
                A = A.to(device)
                label = label.to(device)

                # forward
                output = model(data, A)
                loss = F.cross_entropy(output, label)

                with torch.no_grad():
                    predict = output.argmax(dim=1)
                    num_correct = torch.eq(predict, label).sum().float().item()

                    label_list.append(label.detach())
                    predict_list.append(predict.detach())

                metrics['loss'] += loss.data.cpu().item()
                metrics['num_correct'] += num_correct
                metrics['num_total'] += data.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # statistics
                epoch_samples += 1

            train_description = "\nTrain Epoch [%d|%d]: Loss: %.4f, ACC: %.4f" % (
                epoch, num_epochs,
                metrics['loss'] / epoch_samples,
                metrics['num_correct'] / metrics['num_total']
            )
            self.output_signal.emit(train_description)

            if epoch % 1 == 0:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "epoch_%d.pth" % epoch))

            label_list = torch.cat(label_list, dim=0)
            predict_list = torch.cat(predict_list, dim=0)

            cm = confusion_matrix(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy())
            self.output_signal.emit("confusion_matrix:")
            self.output_signal.emit(str(cm))

            summary_writer.add_scalar(tag="train_loss", scalar_value=metrics['loss'] / epoch_samples, global_step=epoch)
            train_loss = metrics['loss'] / epoch_samples

            # val
            model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            metrics['loss'] = 0
            metrics['num_correct'] = 0
            metrics['num_total'] = 0
            epoch_samples = 0
            label_list = []
            predict_list = []

            for data, A, label in val_dataloader:
                data = data.to(device)
                A = A.to(device)
                label = label.to(device)

                with torch.no_grad():
                    output = model(data, A)
                    loss = F.cross_entropy(output, label)

                    predict = output.argmax(dim=1)
                    num_correct = torch.eq(predict, label).sum().float().item()

                    label_list.append(label.detach())
                    predict_list.append(predict.detach())

                metrics['loss'] += loss.data.cpu().item()
                metrics['num_correct'] += num_correct
                metrics['num_total'] += data.shape[0]

                # statistics
                epoch_samples += 1

            val_description = "Val Epoch [%d|%d]: Loss: %.4f, ACC: %.4f" % (
                epoch, num_epochs,
                metrics['loss'] / epoch_samples,
                metrics['num_correct'] / metrics['num_total']
            )
            self.output_signal.emit(val_description)

            label_list = torch.cat(label_list, dim=0)
            predict_list = torch.cat(predict_list, dim=0)

            cm = confusion_matrix(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy())
            self.output_signal.emit("confusion_matrix:")
            self.output_signal.emit(str(cm))

            summary_writer.add_scalar(tag="val_loss", scalar_value=metrics['loss'] / epoch_samples, global_step=epoch)
            val_loss = metrics['loss'] / epoch_samples

            summary_writer.add_scalars(main_tag="loss", tag_scalar_dict={
                "train": train_loss,
                "val": val_loss
            }, global_step=epoch)

            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if epoch_loss < best_loss:
                #print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_dir, "epoch_best.pth"))

        self.output_signal.emit('Best val loss: {:4f}'.format(best_loss))
        self.output_signal.emit('\nTraining finished')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def main_train(self, data_path, num_epochs, save_dir, seed):
        self.setup_seed(seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dim_hidden = self.settings.get("dim_hidden", 16)
        model_type = self.settings.get("model_type", 1)
        attention_num = self.settings.get("graph_size", 300)
        
        if model_type == 0:
            model = GCN(dim_in=34, dim_hidden=dim_hidden, dim_out=2)
        else:
            model = AttentionGCN(dim_in=34, dim_hidden=dim_hidden, dim_out=2, attention_num=attention_num)

        model = model.to(device)

        not_attack_list = self.settings.get("not_attack_list", [1, 2, 3, 5, 6])
        train_dataset = MyGCNDataset(data_path=data_path, split="train", not_attack_list=not_attack_list, attack_list=[4])
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_dataset = MyGCNDataset(data_path=data_path, split="val", not_attack_list=not_attack_list, attack_list=[4])
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Observe that all parameters are being optimized
        lr = self.settings.get("lr", 0.01)
        weight_decay = self.settings.get("weight_decay", 0.0)
        step_size = self.settings.get("step_size", 10)
        gamma = self.settings.get("gamma", 0.1)

        optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

        model = self.train_model(model, train_dataloader, val_dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, save_dir=save_dir)
