import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

from models.MNIST_LeNet5 import LeNet5
import utils
from utils import train, test


# Global variables
TRAIN_BATCH_SIZE = 128
EPOCHS = 10
CHECKPOINT_INTERVAL = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data'
SAVE_DIR = './save'
MODEL_NAME = 'MNIST_LeNet5'
KEYWORD = MODEL_NAME
CLIPPING = False
if CLIPPING:
    KEYWORD += '_clipped_grads'


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model and optionally append a job_id to the CSV output file.')
    parser.add_argument('--job_id', type=str, default='', help='Optional job ID to append to the CSV file.')
    args = parser.parse_args()

    # record start time
    start_time = datetime.now()
    formatted_now = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"This script was run at {formatted_now}.")
    
    # Generate CSV filename with optional job_id
    csv_filename = f'{MODEL_NAME}'
    if args.job_id:
        csv_filename += f'_{args.job_id}'
    csv_filename += '.csv'

    # tensor transforms pipeline
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST statistics
    ])

    # datasets
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    print(f"Datasets:\n\t-> TRAIN: {'available' if len(train_dataset) > 0 else 'N/A'}\n\t-> TEST: {'available' if len(test_dataset) > 0 else 'N/A'}")

    # data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    print(f"Datasets:\n\t-> TRAIN: {len(train_loader.dataset)} samples\n\t-> TEST: {len(test_loader.dataset)} samples")

    # training loop
    rounds = len(train_loader) * EPOCHS
    print(f"Rounds: {rounds}, Batch Size: {train_loader.batch_size}")
    
    
    model = LeNet5().to(DEVICE)
    print("Model initialized to device:", DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    
    # local variables
    test_losses = []
    train_losses = []
    accuracies = []
    gradients = []
    
    checkpoint_dict = utils.load_checkpoint(keyword=KEYWORD,
                                            save_dir=SAVE_DIR)
    if checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        accuracies = checkpoint_dict['accuracies']
        train_losses = checkpoint_dict['train_losses']
        test_losses = checkpoint_dict['test_losses']
        start_round = checkpoint_dict['round_num'] + 1
        gradients = checkpoint_dict['gradients']
    else:
        start_round = 1
    for round in range(start_round, 1 + rounds):
        print(f"Step: [{round}/{1 + rounds}]")
        batch_idx = (round % len(train_loader)) % len(train_loader)
        updates, loss, grad_norm = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            batch_idx=batch_idx,
            device=DEVICE,
            clipping=CLIPPING)
        
        avg_test_loss, accuracy = test(model = model, 
                                       device = DEVICE, 
                                       test_loader = test_loader, 
                                       criterion=criterion)

        # for plotting
        train_losses.append(float(loss))
        test_losses.append(avg_test_loss)
        accuracies.append(accuracy)
        gradients.append(grad_norm)
        
        # Save checkpoint after every x rounds
        if (round % CHECKPOINT_INTERVAL) == 0:
            utils.save_checkpoint(round_num = round, 
                        model = model, 
                        optimizer = optimizer, 
                        save_dir = SAVE_DIR, 
                        variables_dict = {
                            'test_losses' : test_losses,
                            'train_losses' : train_losses,
                            'accuracies' : accuracies,
                            'gradients' : gradients,
                            },
                        keyword=KEYWORD)
        
        df = pd.DataFrame(data = {
            'Round' : list(range(1, len(train_losses)+1)),
            'train_losses' : train_losses,
            'test_losses' : test_losses,
            'accuracies' : accuracies,
            'gradients' : gradients
        })
        
        df.to_csv(csv_filename, index=False)
    
    execution_time = datetime.now() - start_time
    print("Execution Time: ", execution_time)
    
