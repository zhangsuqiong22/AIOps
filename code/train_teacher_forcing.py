from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import EmDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def combined_loss(pred, target, alpha=0.8):
    mse = nn.MSELoss()(pred, target)
    
    # 计算差分损失
    diff_pred = pred[1:] - pred[:-1]  
    diff_target = target[1:] - target[:-1]
    diff_loss = nn.MSELoss()(diff_pred, diff_target)
    
    return alpha * mse + (1-alpha) * diff_loss


def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)

    model = Transformer().double().to(device)
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)

    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target in dataloader: # for each data set 
        
            optimizer.zero_grad()

            if _input.dim() == 2:
                _input = _input.unsqueeze(0)  # 形状变为 [1, input_length, feature]
            if target.dim() == 2:
                target = target.unsqueeze(0)  # 形状变为 [1, forecast_window, feature]

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            src = _input.permute(1, 0, 2).double().to(device)[:-1, :, :]  # [input_length-1, batch, feature]
            target = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # [input_length-1, batch, feature]

            prediction = model(src, device)  # 模型输出形状: [input_length-1, batch, feature]


            loss = combined_loss(prediction, target[:, :, 0].unsqueeze(-1))
            #loss = criterion(prediction, target[:, :, 0].unsqueeze(-1))  # 仅对第 0 列计算损失
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
    
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"

        # 每 100 个 epoch 可视化一次
        if epoch % 100 == 0:
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            for i in range(src.size(1)):  
                core_idx = i + 1
                #scaler = load(f'scalar_item_feature0.joblib')
                scaler = load(f'scalar_item_server1.joblib')
                src_em_load = scaler.inverse_transform(src[:, i, 0].unsqueeze(-1).cpu())
                target_em_load = scaler.inverse_transform(target[:, i, 0].unsqueeze(-1).cpu())
                prediction_em_load = scaler.inverse_transform(prediction[:, i, 0].detach().unsqueeze(-1).cpu().numpy())

                plot_training(epoch, path_to_save_predictions, src_em_load, prediction_em_load, core_idx, index_in, index_tar)

        train_loss /= len(dataloader)
        scheduler.step(train_loss)

        log_loss(train_loss, path_to_save_loss, train=True)

    plot_loss(path_to_save_loss, train=True)
    return best_model

