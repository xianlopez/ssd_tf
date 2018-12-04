import numpy as np
import cv2
import tools
import TrainEnv
from predict_config import UpdatePredictConfiguration

fps = 30
videoPath = r'T:\Pel Xian\vid_2239.mp4'
# videoPath = r'T:\Pel Xian\Alella_Cam7_personesicotxes_0053.avi'

args = UpdatePredictConfiguration()

assert args.batch_size == 1, 'Batch size must be 1'

net = TrainEnv.TrainEnv(args, 'interactive')

net.start_interactive_session(args)

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
    predictions = net.forward_batch(batch, args)

    # print('predictions')
    # print(len(predictions))
    # print(predictions)

    # Plot result:
    orig_height, orig_width, _ = frame.shape
    # results = tools.convert_boxes_to_original_size(predictions[0], orig_width, orig_height, net.input_shape[0], net.input_shape[1])
    results = predictions[0]
    tools.draw_result(frame, results, net.classnames, None)
    cv2.imshow('result', frame)
    cv2.waitKey(int(1000 / fps))




