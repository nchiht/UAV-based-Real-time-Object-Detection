import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm  # Changed from tqdm.notebook to tqdm
from lib.gan_networks import define_G
import torchvision.transforms as transforms
from sys import path
path.append('./')
from _constants import *



def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
            transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
            transforms.InterpolationMode.NEAREST: Image.NEAREST,
            transforms.InterpolationMode.LANCZOS: Image.LANCZOS}
    return mapper[method]

def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def get_transform(load_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    transform_list = [transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
    else: 
        image_numpy = input_image
    return image_numpy.astype(imtype)

def create_model_and_transform(pretrained=None):
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'resnet_9blocks'
    norm = 'instance'
    no_dropout = True
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []
    
    netG_A = define_G(input_nc, output_nc, ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)
    if pretrained:
        chkpntA = torch.load(pretrained)
        netG_A.load_state_dict(chkpntA)
    netG_A.eval()
    
    # Creating transform
    load_size = 1280
    crop_size = 224
    image_transforms = get_transform(load_size=load_size, crop_size=crop_size)
    return netG_A, image_transforms

def run_inference(img_path, model, transform):
    image = Image.open(img_path)
    inputs = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        out = model(inputs)
    out = tensor2im(out)
    return Image.fromarray(out)

if __name__ == '__main__':
    gan, image_transforms = create_model_and_transform(pretrained=pretrained_model_rainy)

    images_path = set_weather
    save_folder = folder_weather

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for img in os.listdir(images_path):
        trg = os.path.join(images_path, img)
        src = os.path.join(save_folder, img.split('.')[0] + "_rgan.jpg")

        if not os.path.exists(src):
            out = run_inference(img_path=trg, model=gan, transform=image_transforms)
            out.save(src)
