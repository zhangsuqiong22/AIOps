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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, best_model):

    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():

        model.eval()
        for plot in range(25):

            for index_in, index_tar, _input, target in dataloader:
                
                if _input.dim() == 2:
                    _input = _input.unsqueeze(0)  # 添加批次维度 [1, seq_len, features]
                if target.dim() == 2:
                    target = target.unsqueeze(0)  # 添加批次维度 [1, seq_len, features]
                
                src = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # [seq_len-1, batch, features]
                target = target.permute(1, 0, 2).double().to(device)  # [forecast_window, batch, features]

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):
                    
                    prediction = model(next_input_model, device) 
                    
                    if all_predictions == []:
                        all_predictions = prediction
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0)))
                    
                    if next_input_model.size(2) < 2:
                        logger.error(f"特征维度太小: {next_input_model.size()}")
                        break
                        
                    pos_encoding_old_vals = src[i+1:, :, 1:] # 特征列除了第一列
                    
                    new_idx = min(i + 1, target.size(0) - 1)
                    pos_encoding_new_val = target[new_idx, :, 1:].unsqueeze(0)
                    
                    if pos_encoding_old_vals.size(2) == pos_encoding_new_val.size(2):
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val))
                        
                        # 从 src 提取第一个特征，从 prediction 提取预测
                        if i+1 < src.size(0):
                            next_val_feature0 = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0)))
                            
                            if next_val_feature0.size(2) == 1 and pos_encodings.size(2) == next_input_model.size(2) - 1:
                                # 拼接以构建下一个输入
                                next_input_model = torch.cat((next_val_feature0, pos_encodings), dim=2)
                            else:
                                logger.error(f"特征维度不匹配: {next_val_feature0.size()}, {pos_encodings.size()}")
                                break
                        else:
                            logger.error(f"索引 {i+1} 超出范围 {src.size(0)}")
                            break
                    else:
                        logger.error(f"位置编码维度不匹配: {pos_encoding_old_vals.size()}, {pos_encoding_new_val.size()}")
                        break

                if len(all_predictions) > 0:
                    if src.size(0) > 1 and all_predictions.size(0) > 0:
                        true_values = torch.cat((src[1:,:,0], target[:min(target.size(0), all_predictions.size(0)-src.size(0)+1),:,0]))
                        pred_values = all_predictions[:,:,0]
                        
                        min_len = min(true_values.size(0), pred_values.size(0))
                        loss = criterion(true_values[:min_len], pred_values[:min_len])
                        val_loss += loss
            
            val_loss = val_loss / 10 if val_loss > 0 else 0
            
            try:
                for i in range(src.size(1)):
                    core_idx = i + 1
                    scaler = load('scalar_item_feature0.joblib')
                    src_em_load = scaler.inverse_transform(src[:,i,0].unsqueeze(-1).cpu())
                    target_em_load = scaler.inverse_transform(target[:,i,0].unsqueeze(-1).cpu())
                    if all_predictions.size(0) > 0:
                        prediction_em_load = scaler.inverse_transform(all_predictions[:,i,0].detach().unsqueeze(-1).cpu().numpy())
                        plot_prediction(plot, path_to_save_predictions, src_em_load, target_em_load, prediction_em_load, core_idx, index_in, index_tar)
            except Exception as e:
                logger.error(f"可视化过程中出错: {e}")

        logger.info(f"Loss On Unseen Dataset: {val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss}")