import torch
import models.networks
import yaml
import numpy as np
import data.deblur_dataset

with open('./config/deblur_solver.yaml', 'r') as f:
    config = yaml.load(f)

generator, discr = models.networks.get_nets(config["model"])
generator.load_state_dict(torch.load('best_fpn_fpn_mobilenet_content0.5_feature0.006_adv0.001.h5')['model'])
dataset = data.deblur_dataset.DeblurDataset(config, "train")
item = dataset.__getitem__(1)
item = item["A"]
item = np.expand_dims(item, axis=0)
generator = generator.cuda()
generator(item)