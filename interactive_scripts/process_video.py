import numpy as np
import cv2
import tools
import InteractiveEnv
import os
from config.interactive_config_base import InteractiveConfiguration
from ssd import SSDConfig

fps = 30
videoName = 'terminator.mp4'
videoPath = os.path.join(tools.get_base_dir(), 'videos', videoName)

opts = InteractiveConfiguration()
opts.weights_file = r'C:\development\ssd_tf\weights\ssd_training_2019_01_10\model-240'

ssd_config = SSDConfig()

net = InteractiveEnv.InteractiveEnv(opts, ssd_config)

net.start_interactive_session()

cap = cv2.VideoCapture(videoPath)

while True:
    # Capture frame-by-frame:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and batch:
    frame_prep = net.reader.preprocess_image(frame)
    batch = np.expand_dims(frame_prep, axis=0)

    # Forward:
    predictions = net.forward_batch(batch)

    # Plot result:
    orig_height, orig_width, _ = frame.shape
    results = predictions[0]
    tools.draw_result(frame, results, net.classnames, None)
    cv2.imshow('result', frame)
    cv2.waitKey(int(1000 / fps))




