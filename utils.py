import os
import glob
import torch

def save_checkpoint(round_num : int, 
                    model, 
                    optimizer, 
                    save_dir : str, 
                    variables_dict : dict ={}, 
                    keyword : str = None) -> None:
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
        
    # Construct filename with optional keyword
    if keyword:
        filename = os.path.join(save_dir, f'checkpoint_{keyword}_round_{round_num}.pth')
    else:
        filename = os.path.join(save_dir, f'checkpoint_round_{round_num}.pth')

    # Save the checkpoint
    variables_dict.update({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'round_num': round_num})
    
    torch.save(variables_dict, filename)

def load_checkpoint(save_dir : str, 
                    keyword : str = None) -> dict:
    # Search pattern for checkpoint files
    if keyword:
        search_pattern = f'checkpoint_{keyword}_*.pth'
    else:
        search_pattern = 'checkpoint_*.pth'

    checkpoint_files = glob.glob(os.path.join(save_dir, search_pattern))

    if not checkpoint_files:
        print("No checkpoint found, starting from scratch.")
        return None

    # Get the latest checkpoint by creation time
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    # Load the checkpoint
    checkpoint_dict = torch.load(latest_checkpoint)
    print(f"Loaded checkpoint: {latest_checkpoint}")
    
    return checkpoint_dict

def get_batch_by_idx(train_loader, 
                     batch_idx : int) -> ():
    # Calculate start and end indices for the batch
    start_idx = batch_idx * train_loader.batch_size
    end_idx = start_idx + train_loader.batch_size
    
    if end_idx > len(train_loader.dataset):
        end_idx = len(train_loader.dataset)

    # Fetch the data and target tensors for that batch
    data_list, target_list = [], []
    for idx in range(start_idx, end_idx):
        data, target = train_loader.dataset[idx]
        data_list.append(data)
        target_list.append(target)

    # Stack the list into a batch
    data_batch = torch.stack(data_list)
    target_batch = torch.tensor(target_list)

    return data_batch, target_batch