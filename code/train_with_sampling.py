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
import math, random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def flip_from_probability(p):
    return True if random.random() < p else False

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
        for index_in, index_tar, _input, target in dataloader:
        
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            optimizer.zero_grad()
            src = _input.squeeze(0).permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([47, 3, 14])
            target = _input.squeeze(0).permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1.
            sampled_src = src[:1, :, :] #t0 torch.Size([1, 3, 14])

            for i in range(len(target)-1):

                prediction = model(sampled_src, device) # torch.Size([1xw, 3, 1])
                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                if i < 24: # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    ## coin flip
                    v = k/(k+math.exp(epoch/k)) # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(v) # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value

                if prob_true_val: # Using true value as next value
                    sampled_src = torch.cat((sampled_src.detach(), src[i+1, :, :].unsqueeze(0).detach()))
                else: ## using prediction as new value
                    positional_encodings_new_val = src[i+1,:,1:].unsqueeze(0)
                    predicted_em_load = torch.cat((prediction[-1,:,:].unsqueeze(0), positional_encodings_new_val), dim=2)
                    sampled_src = torch.cat((sampled_src.detach(), predicted_em_load.detach()))

            '''
            logger.info(f"Epoch: {epoch}, sampled_src shape: {sampled_src.shape}")  # src shape: torch.Size([47, 1, 7])
            logger.info(f"Epoch: {epoch}, target shape: {target.shape}")   # target shape: torch.Size([47, 1, 7])    
            logger.info(f"Epoch: {epoch}, prediction shape: {prediction.shape}") # prediction shape: torch.Size([46, 1, 1])
            '''
                        
            """To update model after each sequence"""
            loss = criterion(target[:-1,:,0].unsqueeze(-1), prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"


        if epoch % 10 == 0: # Plot 1-Step Predictions
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            for i in range(src.size(1)):
                core_idx = i+1
                scaler = load('scalar_item_core%s.joblib'%core_idx)
                sampled_src_em_load = scaler.inverse_transform(sampled_src[:,i,0].unsqueeze(-1).cpu()) #torch.Size([47, 1, 14])
                src_em_load = scaler.inverse_transform(src[:,i,0].unsqueeze(-1).cpu()) #torch.Size([47, 1, 14])
                target_em_load = scaler.inverse_transform(target[:,i,0].unsqueeze(-1).cpu()) #torch.Size([47, 1, 14])
                prediction_em_load = scaler.inverse_transform(prediction[:,i,0].detach().unsqueeze(-1).cpu().numpy()) #torch.Size([47, 1, 14])
                plot_training_3(epoch, path_to_save_predictions, src_em_load, sampled_src_em_load, prediction_em_load, core_idx, index_in, index_tar)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    plot_loss(path_to_save_loss, train=True)
    return best_model