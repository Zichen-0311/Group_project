import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_length=50):
        self.samples = []
        self.max_length = max_length
        
        for traj in trajectories:
            points = np.array(traj['points'])
            if len(points) < 5:  # Ignore the trajectory that is too short
                continue
                
            # Calculate the relative displacement
            relative_movements = points[1:] - points[:-1]
            
            # Calculate the vector from the starting point to the end point
            start_to_end = points[-1] - points[0]
            distance = np.linalg.norm(start_to_end)
            
            if distance < 1e-6:  # Ignore the motionless trajectory in place
                continue
            
            # Resampling the trajectory
            if len(relative_movements) > self.max_length:
                # If the trajectory is too long, take downsampling.
                indices = np.linspace(0, len(relative_movements)-1, self.max_length, dtype=int)
                relative_movements = relative_movements[indices]
            else:
                #If the trajectory is too short, perform linear interpolation.
                num_points = len(relative_movements)
                x = np.linspace(0, num_points-1, num_points)
                x_new = np.linspace(0, num_points-1, self.max_length)
                relative_movements = np.array([
                    np.interp(x_new, x, relative_movements[:, 0]),
                    np.interp(x_new, x, relative_movements[:, 1])
                ]).T
            
            # Standardized relative mobility
            movement_distances = np.linalg.norm(relative_movements, axis=1)
            mask = movement_distances > 1e-6
            relative_movements[mask] = relative_movements[mask] / movement_distances[mask, np.newaxis]
            
            self.samples.append(relative_movements)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx])

class TrajectoryGenerator(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super(TrajectoryGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden

def train_model(model, train_loader, num_epochs=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Record the training history
    history = {
        'loss': [],
        'main_loss': [],
        'smoothness_loss': [],
        'learning_rate': []
    }
    
    print("Start training...")
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_smoothness_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            #Create input sequences and target sequences
            input_seq = batch[:, :-1, :]
            target_seq = batch[:, 1:, :]
            
            optimizer.zero_grad()
            output, _ = model(input_seq)
            
            # Add smooth loss
            smoothness_loss = torch.mean(torch.norm(output[:, 1:] - output[:, :-1], dim=2))
            
            # Combination reconstruction loss and smoothing loss
            main_loss = criterion(output, target_seq)
            loss = main_loss + 0.1 * smoothness_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_smoothness_loss += smoothness_loss.item()
        
        # Calculate the average loss
        avg_loss = total_loss / len(train_loader)
        avg_main_loss = total_main_loss / len(train_loader)
        avg_smoothness_loss = total_smoothness_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record the history
        history['loss'].append(avg_loss)
        history['main_loss'].append(avg_main_loss)
        history['smoothness_loss'].append(avg_smoothness_loss)
        history['learning_rate'].append(current_lr)
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_trajectory_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 30:
            print("Early stopping triggered")
            break
            
        if (epoch + 1) % 10 == 0:
            time_elapsed = datetime.now() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Main Loss: {avg_main_loss:.4f}, '
                  f'Smoothness Loss: {avg_smoothness_loss:.4f}, '
                  f'LR: {current_lr:.6f}, '
                  f'Time: {time_elapsed}')

    plt.figure(figsize=(15, 10))
    
    #Draw the total loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Draw the main loss
    plt.subplot(2, 2, 2)
    plt.plot(history['main_loss'])
    plt.title('Main Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Draw smoothing loss
    plt.subplot(2, 2, 3)
    plt.plot(history['smoothness_loss'])
    plt.title('Smoothness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Draw the learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    
    # Adjust the layout of the sub-chart
    plt.tight_layout()
    
    # Save the chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_history_{timestamp}.png')
    print(f"训练历史已保存到 training_history_{timestamp}.png")
    
    # Save historical data at the same time
    history_data = {
        'loss': history['loss'],
        'main_loss': history['main_loss'],
        'smoothness_loss': history['smoothness_loss'],
        'learning_rate': history['learning_rate']
    }
    with open(f'training_history_{timestamp}.json', 'w') as f:
        json.dump(history_data, f)
    print(f"The training data has been saved to training_history_{timestamp}.json")
    
    total_time = datetime.now() - start_time
    print(f"The training is done! Total time: {total_time}")
    
    return history

def generate_trajectory(model, start_point, end_point, num_points=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    with torch.no_grad():
        target_vector = end_point - start_point
        distance = np.linalg.norm(target_vector)
        if distance < 1e-6:
            return np.array([start_point])
            
        normalized_vector = target_vector / distance
        
        # Initialize the trajectory
        trajectory = [start_point]
        current_input = torch.zeros((1, 1, 2)).to(device)
        current_input[0, 0] = torch.FloatTensor(normalized_vector).to(device)
        
       # Generate track points
        for _ in range(num_points - 1):
            output, _ = model(current_input)
            delta = output[0, 0].cpu().numpy()
            next_point = trajectory[-1] + delta * (distance / num_points)
            trajectory.append(next_point)
            
            current_direction = (next_point - trajectory[-2]) / np.linalg.norm(next_point - trajectory[-2])
            current_input[0, 0] = torch.FloatTensor(current_direction).to(device)
    
    trajectory = np.array(trajectory)
    
    # Make sure that the starting and ending points of the trajectory are correct
    trajectory[0] = start_point
    trajectory[-1] = end_point
    
    # Use stronger smoothing processing
    smoothed = np.zeros_like(trajectory)
    smoothed[0] = trajectory[0]
    smoothed[-1] = trajectory[-1]
    
    # Multiple smoothing
    for _ in range(3):
        alpha = 0.4
        for i in range(1, len(trajectory)-1):
            smoothed[i] = trajectory[i] * (1-alpha) + (trajectory[i-1] + trajectory[i+1]) * alpha/2
        trajectory = smoothed.copy()
    
    return smoothed

def load_and_prepare_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return TrajectoryDataset(data['trajectories'])

if __name__ == "__main__":
    # Load data
    dataset = load_and_prepare_data('training_data.json')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create and train models
    model = TrajectoryGenerator()
    history = train_model(model, train_loader)
    
    # Save the final model
    torch.save(model.state_dict(), 'trajectory_model.pth') 