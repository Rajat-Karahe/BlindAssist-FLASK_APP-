import io

import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 


def get_model():
	checkpoint_path='currency_checkpoint.pth'
	model=models.mobilenet_v2(pretrained=False)
	model.classifier=nn.Sequential( nn.Dropout(p=0.2, inplace=False),
	                           nn.Linear(1280, 1024),
	                           nn.ReLU(),
	                           nn.Dropout(0.5),
	                           nn.Linear(1024, 512),
	                           nn.ReLU(),
	                           nn.Dropout(0.5),
	                           nn.Linear(512, 9),
	                           nn.LogSoftmax(dim=1))
	model.load_state_dict(torch.load(checkpoint_path,map_location='cpu')['model_state_dict'])
	model.eval()
	return model

def get_tensor(image_bytes):
	my_transforms=transforms.Compose([transforms.Resize(255),
		                              transforms.CenterCrop(224),
		                              transforms.ToTensor(),
		                              transforms.Normalize(
		                              	[0.485,0.456,0.406],
		                              	[0.229,0.224,0.225])])
	image=Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)