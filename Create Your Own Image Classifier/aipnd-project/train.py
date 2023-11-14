import argparse
from model_classifier import *
from load_data import *

parser = argparse.ArgumentParser(description = 'Building Custom model  and inference from the model')

parser.add_argument("data_dir", type=str, help = 'Directory of the images to be trained')
parser.add_argument("--save_dir", type=str, default = 'Checkpoints',  help = 'Directory to save the trained model, the chequepoint')
parser.add_argument("--arch", type=str, default = 'vgg19', help = 'The architecture to be build the model from the list of vgg, resnet, alexnet, or densenet')
parser.add_argument("--learning_rate", type=float, default = 0.003, help = 'learning rate of the model')
parser.add_argument("--hidden_state", type=int, default = 1024, help = 'number of nuerons')
parser.add_argument("--device", type=str, default = 'cuda', help = 'cpu or cuda')
parser.add_argument("--epochs", type=int, default = 4, help = 'Number of epochs to train the model')
parser.add_argument("--drop_p", type=int, default = 0.2, help = 'Probability of dropout')
args = parser.parse_args()



def run_model(data_dir, save_dir, arch,  hidden_state, device, epochs, learning_rate, drop_p):
    trainloader,validloader, testloader, train_data = load_data(data_dir)
    model = classifier()
    model.build_model(arch, learning_rate, hidden_state,drop_p)
    model.train_model(trainloader,validloader, testloader, train_data,save_dir,device,epochs)
    
if __name__ == '__main__':
    run_model(args.data_dir, args.save_dir, args.arch, args.hidden_state, args.device, args.epochs,  args.learning_rate, args.drop_p)