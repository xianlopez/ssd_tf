from config.train_config_base import TrainConfiguration
from DataAugmentation import DataAugOpts
from LRScheduler import LRPolicies, LRSchedulerOpts
import os


class UpdateTrainConfiguration(TrainConfiguration):

    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 1
    batch_size = 16
    optimizer_name = 'momentum'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    ##################################

    lr_scheduler_opts = LRSchedulerOpts(LRPolicies.scheduled)
    lr_scheduler_opts.scheduledPolicyOpts.epochsLRDict = {100: 1e-4, 180: 1e-5}

    dataset_name = 'VOC0712'  # Any folder in the <<root_of_datasets>> directory.

    ##################################
    ######### INITIALIZATION #########
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = os.path.join('.', 'weights', 'vgg_16_for_ssd', 'vgg_16_for_ssd.ckpt')
    modified_scopes = ['conv6', 'conv7', 'conv8', 'conv9', 'scale', 'mbox']
    restore_optimizer = False
    ##################################


    ##################################
    ####### DATA AUGMENTATION ########
    data_aug_opts = DataAugOpts()
    data_aug_opts.apply_data_augmentation = True
    data_aug_opts.write_image_after_data_augmentation = False
    ##################################


    ##################################
    ############ RESIZING ############
    resize_method = 'resize_warp'  # 'resize_warp', 'resize_pad_zeros', 'resize_lose_part', 'centered_crop'
    ##################################

    percent_of_data = 100
    # shuffle_data = False
    # random_seed = 1
    nepochs_save = 50
    nsteps_display = 1
    nepochs_checkval = 50
    nepochs_checktrain = 50
    nonmaxsup = False
    buffer_size = 500
    num_workers = 8
    # root_of_datasets = '/home/xian/datasets'



