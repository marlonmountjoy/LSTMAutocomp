# utilities.py

import torch
from torch.utils.data import DataLoader
#Define the training function, move the model to the GPU or CPU, wrap the training and then the testset in a DataLoader
def train_model(model, device, epochs, batch_size, trainset, testset, optimizer, loss_function, metric='acc'):
    model.to(device)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size)
    #Loop through the dataset for the number of epochs, set the model to training mode, keep track of the total loss
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        #Iterate through the training data in batches, move it to the GPU or CPU, zero the gradients from the last step then;
        #  forward pass, loss function, backwards pass
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
