# Imports here
import torchvision
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from load_data import*
    
class classifier:
    def __init__(self):
        """initializing the atributes of a model"""
        self.drop_p = None
        self.model = None
#         self.model.classifier = None 
        self.criterion = None
        self.optimizer = None
        self.arch = None           
        self.learning_rate = None
        self.epochs = None
        self.device = None            
        self.predict_device = None
        self.save_dir = None          
        self.hidden_state = None
        
    def build_model(self, arch, learning_rate, hidden_state,drop_p):        
        self.arch = arch       
        self.learning_rate = learning_rate
        self.hidden_state = hidden_state
        self.model = getattr(torchvision.models, self.arch)(pretrained=True)    
        for param in self.model.parameters():
            param.requires_grad = False
            
#         if 'vgg' in arch:
#             classifier_input = self.model.classifier[0].in_features   
#         elif 'resnet' in arch:
#             classifier_input = self.model.fc.in_features
#         elif 'alexnet' in arch:
#             classifier_input = self.model.classifier[1].in_features 
#         elif 'densenet' in arch:
#             classifier_input = self.model.classifier.in_features
#         else: 
#             print("please select among the possible model architectures")
            
        classifier = nn.Sequential(nn.Linear(25088, self.hidden_state),
                                        nn.ReLU(),
                                        nn.Dropout(drop_p),
                                        nn.Linear(self.hidden_state, 102),
                                        nn.LogSoftmax(dim=1))
        self.model.classifier  = classifier
#         if 'resnet' in arch:
#             self.model.fc = classifier_part
#             self.optimizer = optim.Adam(self.model.fc.parameters(), lr = self.learning_rate)

#         else: 
#             self.model.classifier  = classifier_part
#             self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.learning_rate)

        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.learning_rate)    
        self.criterion = nn.NLLLoss()
        
    def train_model(self,trainloader,validloader, testloader, train_data, save_dir, device, epochs):
        """A method to train the above defined model by taking the datasets, and save checkpoint by taking the directory"""
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = device
        self.model.to(device)
        epochs = epochs
        steps = 0
        print_every = 5
        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)
                logps = self.model.forward(images)
                loss = self.criterion(logps, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                if steps % print_every == 0:
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
                            
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    self.model.train()
                    
        self.model.class_to_idx = image_datasets[0].class_to_idx
        checkpoint = {'input_size': 25008,
            'output_size': 102,
            'batch_size': 64,            
            'class_to_idx': self.model.class_to_idx,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'model': self.model,
            'classifier': self.model.classifier,
            'state_dict': self.model.state_dict()}
        
        torch.save(checkpoint, self.save_dir + 'checkpoint.pth')
        
    def predict(self, image_path, model, topk, predict_device):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
            '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model = model.to(device)

        image = process_image(image_path)

        image = torch.from_numpy(np.array([image])).float()

        image = image.to(device)

        output = model.forward(image)

        #Top probs
        probs = torch.exp(output).data
        top_p, top_class = probs.topk(topk)

        top_p = top_p.cpu().detach().numpy().tolist()[0]
        top_class = top_class.cpu().detach().numpy().tolist()[0]

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}

        #print(idx_to_class)

        top_classes = [idx_to_class[lab] for lab in top_class]    

        return top_p, top_class

  