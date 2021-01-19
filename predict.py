import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('path', help='Directory of images')
parser.add_argument('checkpoint', help='Checkpoint')
parser.add_argument('--gpu', default=False, help='Use gpu')
parser.add_argument('--top_k', default=5, help='K most likely classes')
parser.add_argument('--category_names', default=None, help='Name and Class')


argums = parser.parse_args()

def load_chpoint(checkpoint):
    chpoint = torch.load(checkpoint)
    model = models.vgg11(pretrained=True)
    
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, int(chpoint['hidden_units'][0]))),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(int(chpoint['hidden_units'][0]), int(chpoint['hidden_units'][1]))),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(p=0.5)),
    ('fc3', nn.Linear(int(chpoint['hidden_units'][1]), 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(chpoint['learning_rate']))
    
    epochs = chpoint['epochs']
    model.class_to_idx = chpoint['class_to_idx']
    model.load_state_dict(chpoint['model_state_dict'])
    optimizer.load_state_dict(chpoint['opt_state_dict'])
    return model

device = torch.device("cuda" if argums.gpu else "cpu")
model = load_chpoint(argums.checkpoint).to(device)

# Image processing
pil_image = Image.open(argums.path)
transform_image = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
np_image = transform_image(pil_image)

np_image = np_image.to(device)
np_image = np_image.unsqueeze(0)
    
model.eval()
with torch.no_grad():
    logps = model.forward(np_image)
    
ps = torch.exp(logps)
probs, top_class_pos = ps.topk(int(argums.top_k), dim=1)
probs, top_class_pos = list(np.array(probs.to('cpu')[0])), np.array(top_class_pos.to('cpu')[0])
pos_to_class = {b: a for a, b in model.class_to_idx.items()}
top_class = [pos_to_class[i] for i in top_class_pos]



with open(argums.category_names, 'r') as f:
    cat_to_name = json.load(f)
print('Top classes: {}'.format([cat_to_name[i] for i in top_class]),
      '\n',
      'Probabilities: {}'.format(probs))