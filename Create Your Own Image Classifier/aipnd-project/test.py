import torch
from model_classifier import load_checkpoint
from load_data import load_data

def test_model(checkpoint_path,data_dir='flowers'):
    _,_, testloader, _ = load_data(data_dir)    
    model = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    criterion = torch.nn.NLLLoss()  # Make sure to define the criterion

    accuracy = 0
    test_loss = 0
    pass_count = 0
    with torch.no_grad():
        for images, labels in testloader:
            pass_count +=1
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            # Calculate the accuracy
            ps = torch.exp(logps)
            top_ps , top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))
     
    print(f"Testing Accuracy: {accuracy/pass_count:.3f}.."
          f"Testing  Loss: {test_loss/len(testloader):.3f}..")

checkpoint_path = 'Models/checkpoint/checkpoint_1.pth'
# checkpoint_path = 'Models/modelcheckpoint.pth'

test_model(checkpoint_path)