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

    del start_idx, end_idx, data_list, target_list
    return data_batch, target_batch

def train(model, 
          train_loader, 
          optimizer, 
          criterion, 
          batch_idx, 
          device,
          clipping = False):
    model.train()
    train_loss = 0
    
    # current weights
    w_t = {name: weights.clone().detach().cpu() for name, weights in model.state_dict().items()}
    
    data, target = get_batch_by_idx(train_loader, batch_idx)
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    # print("input shape:", data.shape)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    # L2 Norm Gradient
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2  # Sum of squares of gradients

    grad_norm = total_grad_norm ** 0.5  # Square root to get the L2 norm
    
    # Gradient CLIPPING
    if clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    train_loss += loss.item()
    
    # new weights
    w_t_plus_1 = {name: weights.clone().detach().cpu() for name, weights in model.state_dict().items()}
    updates = {
        name: (w_t[name] - w_t_plus_1[name])
        for name in w_t_plus_1}
    del w_t_plus_1, train_loss, total_grad_norm, data, target, w_t
    torch.cuda.empty_cache()
    return updates, loss, grad_norm

def test(model, 
         device, 
         test_loader, 
         criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {avg_test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    torch.cuda.empty_cache()
    return avg_test_loss, accuracy

