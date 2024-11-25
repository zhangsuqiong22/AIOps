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

def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
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

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            #src = _input.squeeze(0).double().to(device)[:,:-1,:] # torch.Size([core_nums, 59, 14])
            #src = _input.unsqueeze(0).double().to(device)[:, :-1, :]


            #Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7])
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1.
            #ic(src)  # debug
            #ic(target) # debug
            
            prediction = model(src, device) # torch.Size([core_nums, 59, 1])
            
            #logger.info(f"coreid: {core_idx}, _input shape: {src.shape}, target shape: {target.shape}, prediction shape: {prediction.shape}")
            #ic(prediction) # debug
            
            loss = criterion(prediction, target[:,:,0].unsqueeze(-1))
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()
            

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"


        if epoch % 100 == 0: # Plot 1-Step Predictionsk
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scaler_cpu_load.joblib')
            src_cpu_load = scaler.inverse_transform(src[:,:,0].cpu()) #torch.Size([35, 1, 7])
            target_cpu_load = scaler.inverse_transform(target[:,:,0].cpu()) #torch.Size([35, 1, 7])
            prediction_cpu_load = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            plot_training(epoch, path_to_save_predictions, src_cpu_load, prediction_cpu_load, index_in, index_tar)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
        
        # debug 
        # if epoch == 10:
        #logger.info(f"finish epoch: {epoch}") 
        
    plot_loss(path_to_save_loss, train=True)
    return best_model