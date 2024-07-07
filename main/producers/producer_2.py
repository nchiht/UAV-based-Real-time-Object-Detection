from prod import publish_camera
# from _constants import *

if __name__ == '__main__':
    topic = 'drone_2'
    image_folder = f'data/UAV-benchmark-M/M0201'
    publish_camera(topic, image_folder)