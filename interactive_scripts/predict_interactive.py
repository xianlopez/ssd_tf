import TrainEnv
import os
import cv2
import tools
from interactive_config import UpdateInteractiveConfiguration


def plot_results(predictions, filepaths, net):
    nimages = len(predictions)
    # Loop on images:
    for i in range(nimages):
        results = predictions[i]
        img = cv2.imread(filepaths[i])
        orig_height, orig_width, _ = img.shape
        results = tools.convert_boxes_to_original_size(results, orig_width, orig_height, net.input_shape[0], net.input_shape[1])
        tools.draw_result(img, results, net.classnames, None)
        cv2.imshow('result', img)
        cv2.waitKey(0)


args = UpdateInteractiveConfiguration()

net = TrainEnv.TrainEnv(args, 'interactive')

net.start_interactive_session(args)

for i in range(len(args.image_path)):
    filepaths = [args.image_path[i]]
    input_batch = net.reader.get_batch(filepaths)
    predictions = net.forward_batch(input_batch, args)
    plot_results(predictions, filepaths, net)

