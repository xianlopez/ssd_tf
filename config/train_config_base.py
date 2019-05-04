from ssd import SSDConfig
from mean_ap import MeanAPOpts
from DataAugmentation import DataAugOpts
from LRScheduler import LRPolicies, LRSchedulerOpts
import tools
import os


########### ALL CONFIG ###########
class TrainConfiguration:
    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 5
    batch_size = 32
    optimizer_name = 'sgd'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    vars_to_skip_l2_reg = ['scale', 'biases', 'BatchNorm'] # List with strings contained by the variables that you don't want to add to the L2 regularization loss.
    nbatches_accum = 0 # 0 to not applyl batch accumulation.
    # If train_selected_layers is true, the layers in layers_list are the only ones that are going to be trained.
    # Otherwise, those are the only layers excluded for training.
    # The elements of layers_list do not need to match exactly the layers names. It is enough if they are contained
    # in the layer name. For instance, if we make layers_list = ['fc'] in vgg16, it will include layers fc6, fc7, fc8.
    train_selected_layers = True
    # layers_list = ['fc']  # If this is empy or none, all variables are trained.
    layers_list = []  # If this is empy or none, all variables are trained.
    ##################################


    dataset_name = ''  # Any folder in the <<root_of_datasets>> directory.

    lr_scheduler_opts = LRSchedulerOpts(LRPolicies.onCommand)

    ##################################
    ######### INITIALIZATION #########
    # Weights initialization:
    # To start from sratch, choose 'scratch'
    # To load pretrained weights, and start training with them, choose 'load-pretrained'
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    # To load pretrained weights:
    weights_file = r''
    modified_scopes = []
    restore_optimizer = False
    ##################################


    ##################################
    ####### DATA AUGMENTATION ########
    data_aug_opts = DataAugOpts()
    ##################################


    ##################################
    ############ RESIZING ############
    # Select the way to fit the image to the size required by the network.
    # For DETECTION, use ONLY RESIZE_WARP.
    # 'resize_warp': Resize both sides of the image to the required sizes. Aspect ratio may be changed.
    # 'resize_pad_zeros': Scale the image until it totally fits inside the required shape. We pad with zeros the areas
    #                     in which there is no image. Aspect ratio is preserved.
    resize_method = 'resize_warp'  # 'resize_warp', 'resize_pad_zeros'
    ##################################


    ##################################
    ######## DISPLAYING OPTS #########
    # If recompute_train is false, the metrics and loss shown for a training epoch, are computed with the results
    # obtained with the training batches (thus, not reflecting the performance at the end of the epoch, but during it).
    # Otherwise, we go through all the training data again to compute its loss and metrics. This is more time consuming.
    recompute_train = False
    nsteps_display = 20
    nepochs_save = 100
    nepochs_checktrain = 1
    nepochs_checkval = 1
    ##################################


    mean_ap_opts = MeanAPOpts()


    ##################################
    ####### ONLY FOR DETECTION #######
    th_conf_detection_evaluate = 0.2
    th_conf_detection_predict = 0.6
    # grid_size = 7  # The amount of horizontal (and vertical) cells in which we will divide the image
    threshold_iou_map = 0.5  # Threshold for intersection over union.
    nonmaxsup = True  # Non-maximum supression
    threshold_nms = 0.5  # Non-maximum supression threshold
    ##################################
    # SSD CONFIG
    ssd_config = SSDConfig()
    ##################################
    ##################################



    ##################################
    ########### OTHER OPTS ###########
    percent_of_data = 100  # For debbuging. Percentage of data to use. Put 100 if not debbuging
    num_workers = 8  # Number of parallel processes to read the data.
    root_of_datasets = os.path.join(tools.get_base_dir(), 'datasets')
    experiments_folder = os.path.join(tools.get_base_dir(), 'experiments')
    random_seed = None  # An integer number, or None in order not to set the random seed.
    tf_log_level = 'ERROR'
    buffer_size = 1000 # For shuffling data.
    max_image_size = 600
    gpu_memory_fraction = -1.0
    write_network_input = False
    shuffle_data = True
    ##################################


    ##################################
    ##################################
    # The following code should not be touched:
    outdir = None
