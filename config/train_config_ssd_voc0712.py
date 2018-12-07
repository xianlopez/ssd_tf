from config.train_config_base import TrainConfiguration
from DataAugmentation import DataAugOpts


class UpdateTrainConfiguration(TrainConfiguration):

    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 240
    batch_size = 32
    optimizer_name = 'momentum'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    lr_schedule = {160: 1e-4, 200: 1e-5}
    ##################################


    ##################################
    ######### MODEL AND DATA #########
    loss_name = 'ssdloss' # 'cross-entropy', 'yololoss'
    dataset_name = 'VOC0712'  # Any folder in the <<root_of_datasets>> directory.
    ##################################


    ##################################
    ######### INITIALIZATION #########
    # initialization_mode = 'scratch'  # 'load-pretrained', 'scratch'
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = './weights/vgg_16_for_ssd/vgg_16_for_ssd.ckpt'
    modified_scopes = ['conv6', 'conv7', 'conv8', 'conv9', 'scale', 'mbox']
    ##################################




    ##################################
    ####### DATA AUGMENTATION ########
    data_aug_opts = DataAugOpts()
    data_aug_opts.apply_data_augmentation = True
    data_aug_opts.brightness_prob = 0.5
    data_aug_opts.brightness_delta_lower = -32
    data_aug_opts.brightness_delta_uper = 32
    data_aug_opts.contrast_prob = 0.5
    data_aug_opts.contrast_factor_lower = 0.5
    data_aug_opts.contrast_factor_upper = 1.5
    data_aug_opts.saturation_prob = 0.5
    data_aug_opts.saturation_factor_lower = 0.5
    data_aug_opts.saturation_factor_upper = 1.5
    data_aug_opts.hue_prob = 0.5
    data_aug_opts.hue_delta_lower = -0.1
    data_aug_opts.hue_delta_upper = 0.1
    ##################################


    ##################################
    ############ RESIZING ############
    resize_method = 'resize_warp'  # 'resize_warp', 'resize_pad_zeros', 'resize_lose_part', 'centered_crop'
    ##################################

    nepochs_save = 10
    nepochs_checkval = 5
    nepochs_checktrain = 5
    nsteps_display = 10
    num_workers = 8
    nonmaxsup = True


