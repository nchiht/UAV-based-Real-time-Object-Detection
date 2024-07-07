from prod import publish_camera
from _constants import *

if __name__ == '__main__':
    topic = topic_in_1
    image_folder = f'data/UAV-benchmark-M/{set_producer_1}'
    publish_camera(topic, image_folder)
    