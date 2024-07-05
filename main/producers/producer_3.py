from prod import publish_camera
from _constants import *

if __name__ == '__main__':
    topic = topic_in_3
    # image_folder = f'data/UAV-benchmark-M/{set_weather}_wt'
    image_folder = f'data/UAV-benchmark-M/{set_weather}'
    publish_camera(topic, image_folder)