import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from dataloader import CarDataset
from VAE_v1 import VAE
import torch
from torch import nn
import torch.optim as optim
from sklearn import metrics
import numpy as np

# Could be dice
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

#%% Training

def train_NN(model, train_loader, test_loader, batch_size=64, num_epochs=100, validation_every_steps=500, learning_rate=0.001, loss_fn=nn.CrossEntropyLoss()):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

    step = 0
    model.train()
    
    train_accuracies = []
    valid_accuracies = []
            
    for epoch in range(num_epochs):
        
        train_accuracies_batches = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            output = model(inputs)
            batch_loss = loss_fn(output, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
    
            # Increment step counter
            step += 1
            
            # Compute accuracy.
            predictions = output.max(1)[1]
            train_accuracies_batches.append(accuracy(targets, predictions))
            
            if step % validation_every_steps == 0:
                
                # Append average training accuracy to list.
                train_accuracies.append(np.mean(train_accuracies_batches))
                
                train_accuracies_batches = []
            
                # Compute accuracies on validation set.
                valid_accuracies_batches = []
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets)
    
                        predictions = output.max(1)[1]
    
                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))
    
                    model.train()
                    
                # Append average validation accuracy to list.
                valid_accuracies.append(np.sum(valid_accuracies_batches) / len(test_loader.n_samples))
         
                print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
                print(f"             test accuracy: {valid_accuracies[-1]}")
    
    print("Finished training.")
        
    
    
    
    