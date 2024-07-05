from prod import publish_camera

if __name__ == '__main__':
    topic = 'drone_3'
    image_folder = '/home/cthi/UIT/IE212/UAV-benchmark-M/M0202'
    publish_camera(topic, image_folder)