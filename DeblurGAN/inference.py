import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from models.networks import get_generator
from albumentations import Compose, PadIfNeeded, CenterCrop
import yaml
import os


def inference(args):

    with open(args.config) as cfg:
        config = yaml.load(cfg)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights)['model'])
    model = model.cuda()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    img_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    size_transform = Compose([
        PadIfNeeded(736, 1280)
    ])
    crop = CenterCrop(720, 1280)

    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_s = size_transform(image=img)['image']
    img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
    img_tensor = img_transforms(img_tensor)

    with torch.no_grad():
        img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
        result_image = model(img_tensor)
    result_image = result_image[0].cpu().float().numpy()
    result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
    result_image = crop(image=result_image)['image']
    result_image = result_image.astype('uint8')
    cv2.imwrite(args.output, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test an image')
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--input', required=True, help='Image path')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--weights', required=True, help='Weights path')

    inference(parser.parse_args())
