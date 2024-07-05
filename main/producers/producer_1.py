from prod import publish_camera

if __name__ == '__main__':
    topic = 'drone_1'
    image_folder = '/home/cthi/UIT/IE212/UAV-benchmark-M/M0101'
    publish_camera(topic, image_folder)