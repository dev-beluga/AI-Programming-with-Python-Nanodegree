import argparse
from model_classifier import *
from load_data import *

def input_parse_args():
    parser = argparse.ArgumentParser(description='Train and predict using a deep learning model')
    parser.add_argument("data_dir", type=str, help = 'Directory of the images to be trained')
    parser.add_argument("--save_dir", type=str, default = 'Models',  help = 'Directory to save the trained model, the chequepoint')
    parser.add_argument("--arch", type=str, default='vgg13', choices=['vgg13', 'alexnet', 'resnet50'], help='The architecture to build the model (vgg13, alexnet, or                        resnet50)')
    parser.add_argument("--learning_rate", type=float, default = 0.001, help = 'learning rate of the model')
    parser.add_argument("--hidden_units", type=int, default = 512, help = 'number of hidden inputs')
    parser.add_argument("--device", type=str, default = 'cuda', help = 'cpu or cuda')
    parser.add_argument("--epochs", type=int, default = 1, help = 'Number of epochs to train the model')
    parser.add_argument("--drop_p", type=int, default = 0.2, help = 'Probability of dropout')
    parser.add_argument("--print_every", type=int, default=5, help='Print training and validation information every n steps')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = input_parse_args()
    # Load data and initialize the classifier
    trainloader, validloader, testloader, train_data = load_data(args.data_dir)
    classifier = Classifier()

    # Build the model
    classifier.build_model(args.arch, args.learning_rate, args.hidden_units, args.drop_p)

    # Train the model
    classifier.train_model(trainloader, validloader, train_data, args.save_dir,args.device,args.epochs ,args.print_every)
  