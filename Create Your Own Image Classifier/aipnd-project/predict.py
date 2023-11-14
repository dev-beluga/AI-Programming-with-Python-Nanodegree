from model_classifier import *
from load_data import *
import argparse
import os
import json

def input_parse_args():
    parser = argparse.ArgumentParser(description = 'Inferencing the model , how it predicts')
    parser.add_argument("image_path", type=str, help = 'Input image to be predicted')
    parser.add_argument("check_point", type=str, help = 'Checkpoints folder')
    parser.add_argument("--top_k", type=int, default = 5, help = 'Top probabilities to be displayed')
    parser.add_argument("--category_names", default = None, help = 'JSON file with catergory to name mapped')
    parser.add_argument("--device", type=str,default = 'cpu', help = 'Device type')

    return parser.parse_args()

def get_label_from_path(image_path):
    # Extract the label of the flower from the path name using os.path
    return os.path.split(os.path.dirname(image_path))[1]

def load_category_names(category_names):
    if category_names:
        with open(category_names, 'r') as f:
            return json.load(f)
    return None
                        
def predict(image_path, check_point, top_k, cat_to_name, device):
    # Load the checkpoint                        
    model = load_checkpoint(check_point)
                        
    # Get the real label of the flower                        
    label = get_label_from_path(image_path)
    
    # Predict the classes and their probabilities
    top_p, top_classes = Classifier().predict(image_path, model, device, top_k)
    
    # If category names are provided, map class indices to names
    if cat_to_name:
        cat_to_name = load_category_names(cat_to_name)
        top_class = [cat_to_name[str(i)] for i in top_classes]   
        label = cat_to_name[label]
    else:
        top_class = top_classes 
        
    print(f"The real label :  {label} \n"
          f"The predicted classes and their probabilities are :  {top_class} \n"
          f"{top_p}")
 
if __name__ == '__main__':
    args =  input_parse_args()                       
    predict(args.image_path, args.check_point, args.top_k, args.category_names, args.device)