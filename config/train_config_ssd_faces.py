from ssd import SSDConfig
from DataAugmentation import DataAugOpts
from config.train_config_base import TrainConfiguration


class UpdateTrainConfiguration(TrainConfiguration):

    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 600
    batch_size = 16
    optimizer_name = 'momentum'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    ##################################


    ##################################
    ######### MODEL AND DATA #########
    loss_name = 'ssdloss' # 'cross-entropy', 'yololoss'
    dataset_name = 'Faces'  # Any folder in the <<root_of_datasets>> directory.
    ##################################


    ##################################
    ######### INITIALIZATION #########
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = './weights/pretrained/vgg_16.ckpt'
    modified_scopes = ['conv6', 'conv7', 'block8', 'block9', 'block10', 'block11', 'conv_loc', 'conv_cls', 'scale']
    ##################################


    ssd_config = SSDConfig()
    ssd_config.feat_layers_names = ['block7', 'block8', 'block9', 'block10', 'block11']
    ssd_config.anchor_sizes = [40, 99, 153, 207, 261]
    ssd_config.anchor_ratios = [[1, 2, .5],
                                [1, 2, .5],
                                [1, 2, .5],
                                [1, 2, .5],
                                [1, 2, .5]]
    ssd_config.grid_sizes = [19, 10, 5, 3, 1]

    data_aug_opts = DataAugOpts()
    data_aug_opts.apply_data_augmentation = True
    data_aug_opts.horizontal_flip = True
    data_aug_opts.brightness = 30
    data_aug_opts.contrast = 1.5


    ##################################
    ############ RESIZING ############
    resize_method = 'resize_pad_zeros'  # 'resize_warp', 'resize_pad_zeros', 'resize_lose_part', 'centered_crop'
    ##################################

    percent_of_data = 100
    nepochs_save = 5
    nepochs_checkval = 5

    # root_of_datasets = '/home/xian/datasets'
    nsteps_display = 10
    nonmaxsup = False

    num_workers = 8