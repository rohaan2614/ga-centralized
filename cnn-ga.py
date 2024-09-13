# Ignore all warnings
from ga import GA
import ga_utils
from utils import train, test
import utils
from models.CNN import CNN
import random
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Global variables
TRAIN_BATCH_SIZE = 128
EPOCHS = 10
CHECKPOINT_INTERVAL = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data'
SAVE_DIR = './save'
MODEL_NAME = 'cnn'

CLIPPING = False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Train a model and optionally append a job_id to the CSV output file.')
    parser.add_argument('--job_id', type=str, default='',
                        help='Optional job ID to append to the CSV file.')
    parser.add_argument('--q', type=int, default='10',
                        help='Enter the number of Gaussian vectors. Default = 10')

    args = parser.parse_args()

    q = args.q
    print('q=', q)

    KEYWORD = MODEL_NAME + '_ga' + f'_{TRAIN_BATCH_SIZE}' + f'_{int(q)}'

    if CLIPPING:
        KEYWORD += '_clipped_grads'

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
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_transforms = transforms.Compose([
        transforms.RandomRotation(15), # Randomly rotate the image by 15 degrees
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), # Randomly crop and resize the image
        transforms.RandomAffine(0, translate=(0.1, 0.1)), # Randomly shift image horizontally and vertically
    
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))  # Normalize the images]
        ])

    # No augmentations for test set, just normalize
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # datasets
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transforms)

    print(
        f"Datasets:\n\t-> TRAIN: {'available' if len(train_dataset) > 0 else 'N/A'}\n\t-> TEST: {'available' if len(test_dataset) > 0 else 'N/A'}")

    # data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1000, shuffle=False)

    print(
        f"Datasets:\n\t-> TRAIN: {len(train_loader.dataset)} samples\n\t-> TEST: {len(test_loader.dataset)} samples")

    # training loop
    rounds = len(train_loader) * EPOCHS
    print(f"Rounds: {rounds}, Batch Size: {train_loader.batch_size}")

    model = CNN().to(DEVICE)
    print("Model initialized to device:", DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(lr=0.0003, decay=1e-6)

    # local variables
    test_losses = []
    train_losses = []
    accuracies = []
    deltas = []
    gws = []
    gws_normalized = []
    approximation_errors = []
    gradients = []

    checkpoint_dict = utils.load_checkpoint(keyword=KEYWORD,
                                            save_dir=SAVE_DIR)
    if checkpoint_dict:
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            accuracies = checkpoint_dict['accuracies']
            train_losses = checkpoint_dict['train_losses']
            test_losses = checkpoint_dict['test_losses']
            deltas = checkpoint_dict['deltas']
            gws = checkpoint_dict['gws']
            gws_normalized = checkpoint_dict['gws_normalized']
            approximation_errors = checkpoint_dict['approximation_errors']
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

            # apply GA
            updates_flattened, shapes = ga_utils.flatten_vector(
                updates, device=DEVICE)
            seed = random.randint(0, 100)
            ga = GA(seed=seed,
                    d=len(updates_flattened),
                    q=int(q),
                    device=DEVICE)
            w = ga.w(delta=updates_flattened)
            gw = ga.delta(w=w)

            # Normalize
            gw_normalized = ga_utils.normalize(
                tensor=gw,
                min_val=updates_flattened.min(),
                max_val=updates_flattened.max())

            # calculate error
            error = updates_flattened - gw_normalized
            l2_norm_error = torch.norm(error, p=2)
            approximation_errors.append(l2_norm_error)

            # apply approximated updates
            curr_weights_dict = {
                name: weights.clone().detach().cpu()
                for name, weights in model.state_dict().items()}

            curr_weights, _ = ga_utils.flatten_vector(
                curr_weights_dict, device=DEVICE)
            new_weights = curr_weights + gw_normalized

            new_weights_dict = ga_utils.vector_to_state_dict(flat_vector=new_weights,
                                                            shapes=shapes,
                                                            torchvision_model=True)

            model.load_state_dict(state_dict=new_weights_dict)

            avg_test_loss, accuracy = test(model=model,
                                        device=DEVICE,
                                        test_loader=test_loader,
                                        criterion=criterion)

            # for plotting
            train_losses.append(float(loss))
            test_losses.append(avg_test_loss)
            accuracies.append(accuracy)
            deltas.append(updates_flattened)
            gws.append(gw)
            gws_normalized.append(gw_normalized)
            gradients.append(grad_norm)

            # Save checkpoint after every x rounds
            if (round % CHECKPOINT_INTERVAL) == 0:
                utils.save_checkpoint(round_num=round,
                                    model=model,
                                    optimizer=optimizer,
                                    save_dir=SAVE_DIR,
                                    variables_dict={
                                        'test_losses': test_losses,
                                        'train_losses': train_losses,
                                        'accuracies': accuracies,
                                        'deltas': deltas,
                                        'gws': gws,
                                        'gws_normalized': gws_normalized,
                                        'approximation_errors': approximation_errors,
                                        'gradients': gradients,
                                    },
                                    keyword=KEYWORD)

            df = pd.DataFrame(data={
                'Round': list(range(1, len(train_losses)+1)),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies,
                'gradients': gradients
            })

            df.to_csv(csv_filename, index=False)

        execution_time = datetime.now() - start_time
        print("Execution Time: ", execution_time)
