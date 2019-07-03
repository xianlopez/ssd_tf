# README #

## UNDER CONSTRUCTION

Tensorflow framework to train SSD

It is based on Python 3.5. You will need the following packages:
* tensorflow-gpu==1.11.0
* lxml
* matplotlib==2.2.3
* opencv-python

## Running a pretrained model

Link to download pretrained weights (trained with this code): https://drive.google.com/open?id=1glqEz7LTX9fMAv8m8JAVq_Gekqq_dEuv

Link to sample video: https://drive.google.com/open?id=15hv3YDOR6PGsyGeqsYLyy_lqOEtgHmzo

Place the video in `ssd_tf/videos/terminator.mp4`, and the weights in `ssd_tf\weights\ssd_training_2019_01_10`. Then run `process_video.py`.

## Training

Link to download dataset (VOC0712): https://drive.google.com/file/d/1TtO5FD2g2bmzfyyGsqp7fnNvZi_0Qp-2/view?usp=sharing

Link to download weights (VGG16 weights adapted to start training SSD): https://drive.google.com/open?id=1d4S0aZT7WDoPkKkTANnQKaVFDVsxsw_j

Since I couldn't find an easy and convenient way to automatically download these files, please do it by hand clicking on the links above. After this, extract them and put the weights in `weights/vgg16_for_ssd` and the dataset in `datasets/VOC0712`.

To make a training from scratch (well, starting from the VGG-16 weights), copy the file `train_config_ssd.py` to the root and rename it to `train_config.py`. This can be done by executing the following command:

``cp config/train_config_ssd.py train_config.py``

To start training run

``python main.py -r train``

## To Do
* Understand why with TensorFlow 1.13 the loss is more likely to diverge than with 1.11.
* Show example of training.
* Add other base networks and other resolutions.