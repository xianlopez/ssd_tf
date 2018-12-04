import os
import tools
from config.train_config import TrainConfiguration, PreprocessOpts, YoloConfig, SSDConfig
from DataAugmentation import DataAugOpts


class UpdateTrainConfiguration(TrainConfiguration):

    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 256
    batch_size = 32
    optimizer_name = 'momentum'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    ##################################


    ##################################
    ######### MODEL AND DATA #########
    model_name = 'ssd'  # 'vgg16', 'resnet50', 'mnistnet', 'yolo'
    loss_name = 'ssdloss' # 'cross-entropy', 'yololoss'
    dataset_name = 'VOC_2007_full'  # Any folder in the <<root_of_datasets>> directory.
    ##################################


    ##################################
    ######### INITIALIZATION #########
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = './weights/pretrained/vgg_16.ckpt'
    modified_scopes = ['conv6', 'conv7', 'block8', 'block9', 'block10', 'block11', 'conv_loc', 'conv_cls', 'scale']
    ##################################


    ##################################
    ########### PREPROCESS ###########
    preprocess_opts = PreprocessOpts()
    preprocess_opts.type = 'subtract_mean'
    preprocess_opts.mean = 'vgg'
    ##################################


    data_aug_opts = DataAugOpts()
    data_aug_opts.apply_data_augmentation = True
    data_aug_opts.horizontal_flip = True
    data_aug_opts.brightness = 30
    data_aug_opts.contrast = 1.5
    data_aug_opts.sample_patch_like_ssd = True


    ##################################
    ############ RESIZING ############
    resize_method = 'resize_pad_zeros'  # 'resize_warp', 'resize_pad_zeros', 'resize_lose_part', 'centered_crop'
    ##################################

    percent_of_data = 100
    nepochs_save = 10
    nepochs_checkval = 10
    nepochs_checktrain = 10

    # root_of_datasets = '/home/xian/datasets'
    nsteps_display = 20
    nonmaxsup = True

    num_workers = 8

    write_image_after_data_augmentation = False