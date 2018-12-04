import numpy as np
import cv2
import os
import TrainEnv

dataset_dir = r'D:\datasets\FA_OCC_SH_2'
split = 'val'
files_list = os.path.join(dataset_dir, split + '_labels.txt')

args = UpdatePredictConfiguration()

assert args.batch_size == 1, 'Batch size must be 1'

net = TrainEnv.TrainEnv(args, 'interactive')

net.start_interactive_session(args)

ncorrect = 0

with open(files_list, 'r') as fid:
    lines = fid.read().split('\n')
    total_pairs = len(lines)
    for pair in lines:
        print(pair)
        if pair == '':
            continue
        line_split = pair.split(',')
        path_left = os.path.join(dataset_dir, line_split[0])
        path_right = os.path.join(dataset_dir, line_split[1])
        class_id = int(line_split[2])

        # img_left = cv2.imread(path_left)
        # img_right = cv2.imread(path_right)
        #
        # img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        # img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        #
        # image_left_prep = net.reader.preprocess_image(img_left)
        # image_right_prep = net.reader.preprocess_image(img_right)
        #
        # batch_left = np.expand_dims(image_left_prep, axis=0)
        # batch_right = np.expand_dims(image_right_prep, axis=0)

        img_left = net.reader.get_image(path_left)
        img_right = net.reader.get_image(path_right)
        batch_left = np.expand_dims(img_left, axis=0)
        batch_right = np.expand_dims(img_right, axis=0)

        # Forward:
        predictions = net.forward_tampering_batch(batch_left, batch_right, args)

        print('predicted: ' + str(np.argmax(predictions[0])) + '  -  label: ' + str(class_id))
        if np.argmax(predictions[0]) == class_id:
            ncorrect += 1

        # Plot result:
        if False:
            print(net.classnames[np.argmax(predictions[0])])
            print('predicted: ' + str(predictions[0]) + '  -  label: ' + str(class_id))
            cv2.imshow('left', img_left)
            cv2.imshow('right', img_right)
            cv2.waitKey(0)

accuracy = float(ncorrect) / total_pairs * 100.0

print('ncorrect: ' + str(ncorrect))
print('Accuracy: ' + str(accuracy))


