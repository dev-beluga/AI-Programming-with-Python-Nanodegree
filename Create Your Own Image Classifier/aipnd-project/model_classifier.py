# Imports here
import torchvision
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from load_data import*
import os
import time
    
class Classifier:
    def __init__(self):
        ''' Initial values '''
        self.model = None
        self.arch = None
        self.hidden_units = None
        self.learning_rate = None
        self.epochs = None
        self.drop_p = None        
        self.criterion = None
        self.optimizer = None        
        self.device = None            
        self.save_dir = None     
        self.print_every = None
                
    def build_model(self, arch, learning_rate, hidden_units,drop_p): 
        ''' Build the model '''
        self.arch = arch       
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.drop_p = drop_p
        
        # Load a pre-trained model from torchvision
#         self.model = getattr(torchvision.models, self.arch)(pretrained=True)
        
        # Load a pre-trained model from torchvision based on the user's choice
        if self.arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
            input_size = self.model.classifier[0].in_features
        elif self.arch == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            input_size = self.model.classifier[1].in_features
        elif self.arch == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            input_size = self.model.fc.in_features  # Using fc for resnet50
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        
        # Freeze parameters to avoid backpropagation
        for param in self.model.parameters():
            param.requires_grad = False
           
        if hasattr(self.model, 'classifier'):
            classifier = nn.Sequential(nn.Linear(input_size, self.hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_p),
                                        nn.Linear(self.hidden_units, 102),
                                        nn.LogSoftmax(dim=1))
            self.model.classifier = classifier
        elif hasattr(self.model, 'fc'):
            fc = nn.Sequential(nn.Linear(input_size, self.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(self.drop_p),
                               nn.Linear(self.hidden_units, 102),
                               nn.LogSoftmax(dim=1))
            self.model.fc = fc
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
            
        # Adjust the optimizer initialization based on the model's architecture
        if hasattr(self.model, 'classifier'):
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        elif hasattr(self.model, 'fc'):
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)        
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
         
        self.criterion = nn.NLLLoss()
        
    def train_model(self,trainloader,validloader, train_data, save_dir, device, epochs, print_every):
        ''' Train the loaded  model '''
        self.epochs = epochs
        self.print_every = print_every
        self.save_dir = save_dir
        self.device = device
        self.model.to(device)
        steps = 0
        print_every = 5
        # Record the start time
        start_time = time.time()

        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass and backward pass
                logps = self.model.forward(images)
                loss = self.criterion(logps, labels)                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                if steps % print_every == 0:
                    # Record the time before printing
                    print_start_time = time.time()

                    self.model.eval()
                    valid_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            loss = self.criterion(logps, labels)
                            valid_loss += loss.item()

                            # Calculate the accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    # Calculate and print the elapsed time for print_every batches
                    print_elapsed_time = time.time() - print_start_time                           
                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}.. "
                          f"Time taken for {print_every} batches: {print_elapsed_time:.2f} seconds")
                   
                    running_loss = 0
                    self.model.train()
                    
        # Record the end time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds ...{total_time/60:.2f} minutes \n")
        print(f"Model Trained Successfully ! \n")
        
        # Save the model checkpoint
        self.save_checkpoint(train_data)
        
    def save_checkpoint(self, train_data):
        ''' Save a checkpoint of the trained model '''
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        # Ensure the directory exists, create it if necessary
        os.makedirs(checkpoint_dir, exist_ok=True)
        base_name = 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, base_name)
        # Check if the file already exists
        count = 1
        while os.path.exists(checkpoint_path):
            filename, extension = os.path.splitext(base_name)
            checkpoint_path = os.path.join(checkpoint_dir, f"{filename}_{count}{extension}")
            count += 1
        self.model.class_to_idx = train_data.class_to_idx
        # Determine the input_size based on the model's architecture
        if hasattr(self.model, 'classifier'):
            input_size = self.model.classifier[0].in_features
        elif hasattr(self.model, 'fc'):
            input_size = self.model.fc[0].in_features
    
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        checkpoint = {'input_size': input_size,
                      'output_size': 102,
                      'batch_size': 64,            
                      'class_to_idx': self.model.class_to_idx,
                      'criterion': self.criterion,
                      'optimizer': self.optimizer,
                      'model': self.model,
                      'state_dict': self.model.state_dict()}
        
        # Check if the model has a classifier or fc attribute and save it accordingly
        if hasattr(self.model, 'classifier'):
            checkpoint['classifier'] = self.model.classifier
        elif hasattr(self.model, 'fc'):
            checkpoint['fc'] = self.model.fc
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        torch.save(checkpoint, checkpoint_path)             
        print(f"Checkpoint saved ! \n")    
    

    def predict(self, image_path, model, device, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model. '''
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        
        # Process the image
        image = process_image(image_path)
        image = torch.from_numpy(np.array([image])).float()
        image = image.to(device)
        
        # Forward pass
        output = self.model.forward(image)

        #Top probs
        probs = torch.exp(output).data
        top_p, top_class = probs.topk(topk)

        top_p = top_p.cpu().detach().numpy().tolist()[0]
        top_class = top_class.cpu().detach().numpy().tolist()[0]
        
        # Map indices to class labels
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        top_classes = [idx_to_class[lab] for lab in top_class]    

        return top_p, top_classes