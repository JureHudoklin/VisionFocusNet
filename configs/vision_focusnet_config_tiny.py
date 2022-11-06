import os
import yaml

class Config():
    """
    Configuration of the FocusNet
    """
    ############
    # Training #
    ############
    LR = 0.0001
    LR_BACKBONE = 1e-5
    WEIGHT_DECAY = 0.0001 
    LR_DROP = 25 # Drop LR after X epochs
    
    BATCH_SIZE = 1
    EPOCHS = 50
    SAVE_BEST_ONLY = False
    AUX_LOSS = True  # If we want outputs of all transformer layers --> add loss for each layer
    EOS_COEF = 0.1 # Weight for no-object class
    
    FOCAL_ALPHA = 0.25
    
    CLASS_LOSS_COEF = 1.0
    SIM_LOSS_COEF = 0.1
    BBOX_LOSS_COEF = 5.0
    GIOU_LOSS_COEF = 2.0
    
    
    ############
    # Backbone #
    ############
    BACKBONE = 'resnet34'
    DILATION = False
    POSITION_EMBEDDING = 'sine'
    RETURN_INTERM_LAYERS = True # masks
        
    ###############
    # Transformer #
    ###############
    
    DROPOUT = 0.0
    N_HEADS = 8
    
    NUM_QUERIES = 50 # Num of object queries
    TWO_STAGE = False
    D_MODEL = 256
    DIM_FEEDFORWARD = 2048
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    ACTIVATION = 'prelu'
    QUERY_SCALE_TYPE = 'cond_elewise' # 'cond_scalar'
    MODULATE_HW_ATTN = True
    LOOK_FORWARD_TWICE = True
    
    ####################
    # Template Encoder #
    ####################
    TEMPLATE_ENCODER = {
        "LR" : 5e-6,
        "PRETRAINED" : True,
    }
    
    # DN-DETR
    DN_ARGS = {
        "USE_DN" : True,
        "USE_DN_AUX" : True,
        "LABEL_NOISE_SCALE": 0.2,
        "BOX_NOISE_SCALE": 0.2,
        "NUM_DN_GROUPS": 5,
        "USE_INDICATOR": False,
    }
    
    #################
    # Augmentations #
    #################
    TGT_IMG_SIZE = (256, 256)
    
    
    ###########
    # Matcher #
    ###########
    SET_COST_CLASS = 2.0
    SET_COST_BBOX = 5.0
    SET_COST_GIOU = 2.0

    ###########
    # Dataset #
    ###########
    NUM_WORKERS = 6
    NUM_TGTS = 3
    PIN_MEMORY = True
    COCO_PATH = "/hdd/datasets/COCO/images"
    AVD_PATH = "/home/jure/datasets/AVD/ActiveVisionDataset"
    GMU_PATH = "/home/jure/datasets/GMU_kitchens/data"
    OBJECTS_365_PATH = "/home/jure/datasets/Objects365/data"

    def __init__(self, load_path = None, save_path = None):
        """
        Loads the configuration for the network. If a save or load file is provided the configuration will be opened/saved accordingly.

        Args:
            save_path (str): Path to the configuration file.
            load_path (str): Path to the configuration file.
        """
        # save configs
        config_vars = vars(Config)
        config_dict = {}
        for attr in list(config_vars.keys()):
            if not attr.startswith("__"):
                config_dict[attr] = config_vars[attr]
        print("Configuration is loaded: ")
        print(config_dict)
        
        if load_path is not None:
            with open(os.path.join(load_path, 'config.yml'), 'r') as yaml_file:
                config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for attr in list(config_dict.keys()):
                if not attr.startswith("__"):
                    setattr(Config, attr, config_dict[attr])
            print("Configuration is loaded: ")
            print(config_dict)

        if save_path is not None:
            with open(os.path.join(save_path, 'config.yml'), 'w') as yaml_file:
                yaml.dump(config_dict, yaml_file, default_flow_style=False)

      