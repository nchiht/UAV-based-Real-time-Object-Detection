from prod import publish_camera
from Weather_Effect_Generator.CyclicGAN import run_inference, gan, image_transforms
from tqdm import tqdm
import os

if __name__ == '__main__':
    topic = 'drone_3'
    image_folder = '/home/cthi/UIT/IE212/UAV-benchmark-M/M0202'
    save_folder = '/home/cthi/UIT/IE212/images_with_weather/M0202_wt'

    for img in tqdm(image_folder):
        trg = image_folder + '/' + img
        src = os.path.join(save_folder, img.split('.')[0] + "-sgan.jpg")

        if not os.path.exists(src):
            os.makedirs(src)

        out = run_inference(img_path=trg, model=gan, transform=image_transforms)
        out.save(src)


    publish_camera(topic, save_folder)