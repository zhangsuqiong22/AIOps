import argparse
import re
from train_teacher_forcing import *
#from train_with_sampling import *
#from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *


def move_best_model(best_model, path_to_save_model, best_model_save_path):
    if not os.path.exists(best_model_save_path):
        os.makedirs(best_model_save_path)
    source_model_path = os.path.join(path_to_save_model, best_model)
    destination_model_path = os.path.join(best_model_save_path, best_model)
    shutil.move(source_model_path, destination_model_path)
    new_model_name = re.sub(r'_\d+', '', best_model)  # remove index e.g.,"_603"
    new_model_path = os.path.join(best_model_save_path, new_model_name)
    os.rename(destination_model_path, new_model_path)

def main(
    epoch: int = 10000,
    k: int = 60,
    option: str = "inference",
    batch_size: int = 1,
    frequency: int = 100,
    training_length = 60,
    #training_length = 11,
    forecast_window = 12,
    #train_csv = "train_dataset_20240905.csv",
    train_csv = "june.csv",
    #train_csv = "train_dataset_merged.csv",
    #train_csv = "train_dataset_core123sum.csv",
    test_csv = "june.csv",
    path_to_save_model = "save_model/",
    path_to_save_loss = "save_loss/", 
    path_to_save_predictions = "save_predictions/",
    device = "cpu",
    tflite_inference = False
):

    clean_directory()

    #train_dataset = EmDataset(csv_name = train_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    #train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    ## train_teacher_forcing
    #best_model = transformer(train_dataloader, epoch, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    
    ## train_with_sampling
    ## best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)

    ##test_dataset = EmDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    ##test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
    ##inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)


    if option == "train":
        print("Starting training...")

        # Data preparation for training
        train_dataset = EmDataset(
            csv_name = train_csv,
            root_dir = "Data/",
            length = training_length,
            forecast_window = forecast_window,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Training mode: Call the transformer model training function
        best_model = transformer(
            train_dataloader, args.epoch, k, args.frequency,
            args.path_to_save_model, args.path_to_save_loss,
            args.path_to_save_predictions, args.device
        )

        # Save the best model to the specified path
        move_best_model(best_model, path_to_save_model, args.best_model_save_path)
        print(f"Training complete. Best model saved at {args.best_model_save_path}")

    elif option == "inference":
        print("Starting inference...")

        # Data preparation for inference
        test_dataset = EmDataset(
            csv_name = test_csv,
            root_dir = "Data/",
            length = training_length,
            forecast_window = forecast_window,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # Inference mode: Use the pre-trained model to make predictions
        #best_model_path = "temp_model/test.pth"  # Modify this path as needed for your model
        best_model_path = os.path.join(args.best_model_save_path, args.model_name)
        predictions = inference(
            path_to_save_predictions,  
            forecast_window,            
            test_dataloader,                 
            args.device,                     
            best_model_path,
            tflite_inference
        )
        print(f"Inference complete. Predictions saved at {args.path_to_save_predictions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default="save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default="save_predictions/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--best_model_save_path",type=str,default="best_model/")
    parser.add_argument("--model_name", type=str, default="best_train.pth", help="Name for the best saved model.")
    parser.add_argument("--tflite_inference", type=str, default=None)
    parser.add_argument(
        "--option", 
        type=str,
        help='train: training mode; inference: inference mode', 
        choices=('train', 'inference'), 
        default="inference"
    )
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        device=args.device,
        option=args.option,
        tflite_inference=args.tflite_inference
    )

