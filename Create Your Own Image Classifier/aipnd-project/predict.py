from model_classifier import *
from load_data import *
import argparse
import re
import json


parser = argparse.ArgumentParser(description = 'predict by loading the model')

parser.add_argument("image_path", type=str, help = 'directory of the image to be predicted')
parser.add_argument("check_point", type=str, help = 'directory for saved checkpoint')
parser.add_argument("--top_k", type=int, default = 5, help = 'how many classes to show')
parser.add_argument("--category_names", default = None, help = 'json file containing label and corresponding names')
parser.add_argument("--device", type=str,default = 'cpu', help = 'cuda or cpu')

args = parser.parse_args()

def predict(image_path, check_point, top_k, cat_to_name, device):
    model = load_checkpoint(check_point)  #loading the checkpoint 
    label =re.search('(?<=\/)([0-9]*)\/', image_path).group(1)  #to get the label of the flower from the path name
    
    prop, cls = classifier().predict(image_path, model, top_k, device)
    if cat_to_name:
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        cls = [cat_to_name[str(i)] for i in cls]   
        label = cat_to_name[label]

    
    print('The real label/name of the flower is : ', label)
    print('The predicted classes and their probabilitis are : ', cls, prop)
 
if __name__ == '__main__':
    predict(args.image_path, args.check_point, args.top_k, args.category_names, args.device)