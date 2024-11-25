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
import subprocess, os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def tflite_model_execute(input, model_path):
    input_bin = "TFLite/testapp/data/input.bin"
    output_bin = "TFLite/testapp/data/output.bin"

    if os.path.exists(input_bin):
        os.remove(input_bin)
    if os.path.exists(output_bin):
        os.remove(output_bin)

    with open(input_bin, 'wb') as f:
        numpy_array = input.float().numpy()
        f.write(numpy_array.tobytes())
    
    exe_path = "TFLite/tools/AiMlApp"
    args = [model_path, input_bin, output_bin]
    # Load TFLite model and allocate tensors.
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)

    with open(output_bin, 'rb') as f:
        output_org = np.frombuffer(f.read(), dtype=np.float32)
        output_org = output_org.copy()
        output = torch.from_numpy(output_org).reshape(3, 59, 1)

    return output


def inference(path_to_save_predictions, forecast_window, dataloader, device, best_model, tflite_model=None):
    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(best_model, weights_only=True))
    criterion = torch.nn.MSELoss()

    if tflite_model:
        print("Using TFLite model and C++ inference engine for inference")

    val_loss = 0
    with torch.no_grad():

        model.eval()

        for plot in range(25):

            for index_in, index_tar, _input, target in dataloader:
                
                # starting from 1 so that src matches with target, but has same length as when training
                src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                target = target.permute(1,0,2).double().to(device) # t48 - t59

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):
                    
                    prediction = model(next_input_model, device) # 47,1,1: t2' - t48'

                    if all_predictions == []:
                        all_predictions = prediction # 47,1,1: t2' - t48'
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                    pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                    
                    next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'
                    next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round

                true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                loss = criterion(true, all_predictions[:,:,0])
                val_loss += loss
            
            val_loss = val_loss/10
            scaler = load('scaler_cpu_load.joblib')
            src_cpu_load = scaler.inverse_transform(src[:,:,0].cpu())
            target_cpu_load = scaler.inverse_transform(target[:,:,0].cpu())
            prediction_cpu_load = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
            plot_prediction(plot, path_to_save_predictions, src_cpu_load, target_cpu_load, prediction_cpu_load, index_in, index_tar)

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")