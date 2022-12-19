import os
import yaml

class Config():
    """
    Configuration of the FocusNet
    """
    ############
    # Training #
    ############
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    TRAIN_METHOD = "both" # "contrastive_only", "detection_only", "both"
    LR_DROP = 25 # Drop LR after X epochs
    MAX_NORM = 0.1
    
    BATCH_SIZE = 9
    EPOCHS = 50
    SAVE_BEST_ONLY = False
    AUX_LOSS = True  # If we want outputs of all transformer layers --> add loss for each layer
    EOS_COEF = 0.1 # Weight for no-object class
    FOCAL_ALPHA = 0.25
    
    ################
    # LOSS WEIGHTS #
    ################
    LOSS_WEIGHTS = {
        "loss_ce": 1.0, # Classification loss for same objects 
        "loss_sim" : 1.0, # Classification loss for similar objects
        "loss_bbox" : 5.0, # Bounding box regression loss weight
        "loss_giou" : 2.0, # Generalized IoU loss weight
    }
    CONTRASTIVE_LOSS = 0.2
    CENTEREDNESS_LOSS = 1.0
    
    ############
    # Backbone #
    ############
    BACKBONE = {
        "name": "resnet50",
        "dilation": False,
        "return_intermediate_layers": True,
        "lr" : 1e-5,
        "pretrained": True,
    }
    POSITION_EMBEDDING =  "sine"

    
    ###############
    # Transformer #
    ###############
    
    DROPOUT = 0.0
    N_HEADS = 8
    NUM_LEVELS = 2
    NUM_QUERIES = 100 # Num of object queries
    TWO_STAGE = False
    D_MODEL = 256
    DIM_FEEDFORWARD = 2048
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    ACTIVATION = 'prelu'
    QUERY_SCALE_TYPE = 'cond_elewise' # 'cond_scalar'
    MODULATE_HW_ATTN = True
    LOOK_FORWARD_TWICE = True
    
    ####################
    # Template Encoder #
    ####################
    TEMPLATE_ENCODER = {
        "name": "vits16",
        "pretrained" : True,
        "lr" : 0, #2e-6,
        "use_checkpointing" : True,
    }
    
    # DN-DETR
    DN_ARGS = {
        "USE_DN" : True,
        "USE_DN_AUX" : True,
        "LABEL_NOISE_SCALE": 0.2,
        "BOX_NOISE_SCALE": 0.3,
        "NUM_DN_GROUPS": 5,
        "USE_INDICATOR": False,
    }
    
    #################
    # Augmentations #
    #################
    TGT_IMG_SIZE = 224
    TGT_MAX_IMG_SIZE = 448
    
    ###########
    # Matcher #
    ###########
    SET_COST_CLASS = 1.0
    SET_COST_BBOX = 5.0
    SET_COST_GIOU = 2.0

    ###########
    # Dataset #
    ###########
    NUM_WORKERS = 2
    NUM_TGTS = 5
    TGT_MIN_AREA = 500
    PIN_MEMORY = True
    
    COCO_PATH = "/home/jure/datasets/COCO/images"
    OBJECTS_365_PATH = "/home/jure/datasets/Objects365/data"
    
    AVD_TRAIN = "/home/jure/datasets/AVD/avd_train_coco_gt.json"
    AVD_VAL = "/home/jure/datasets/AVD/avd_val_coco_gt.json"
    AVDSUP_VAL = "/home/jure/datasets/AVD/avdsup_val_coco_gt.json"
    
    GMU_TRAIN = "/home/jure/datasets/GMU_kitchens/gmu_train_coco_gt.json"
    GMU_VAL = "/home/jure/datasets/GMU_kitchens/gmu_val_coco_gt.json"
    GMUSUP_VAL = "/home/jure/datasets/GMU_kitchens/gmusup_val_coco_gt.json"
    
    TLESS_TRAIN = "/home/jure/datasets/T-LESS/tless_train_coco_gt.json"
    TLESS_VAL = "/home/jure/datasets/T-LESS/tless_val_coco_gt.json"
    TLESSSUP_VAL = "/home/jure/datasets/T-LESS/tlesssup_val_coco_gt.json"
    
    YCBV_TRAIN = "/home/jure/datasets/ycbv_processed/ycbv_train_coco_gt.json"
    YCBV_VAL = "/home/jure/datasets/ycbv_processed/ycbv_val_coco_gt.json"
    YCBVSUP_VAL = "/home/jure/datasets/ycbv_processed/ycbvsup_val_coco_gt.json"
    
    ICTR_TRAIN = "/home/jure/datasets/icbin/train/icbintrain_train_coco_gt.json"
    ICTR_VAL = "/home/jure/datasets/icbin/train/icbintrain_val_coco_gt.json"
    ICTRSUP_VAL = "/home/jure/datasets/icbin/train/icbintrainsup_val_coco_gt.json"
    
    ICVAL_TRAIN = "/home/jure/datasets/icbin/val/icbinval_train_coco_gt.json"
    ICVAL_VAL = "/home/jure/datasets/icbin/val/icbinval_val_coco_gt.json"
    ICVALSUP_VAL = "/home/jure/datasets/icbin/val/icbinvalsup_val_coco_gt.json"
   
    ICBIN_TRAIN_PATH = "/home/jure/datasets/icbin/train"
    ICBIN_VAL_PATH = "/home/jure/datasets/icbin/val"
    
    LM_TRAIN_PATH = "/home/jure/datasets/LM/synthetic/lm_train_coco_gt.json"
    LM_VAL_PATH = "/home/jure/datasets/LM/val/lm_val_coco_gt.json"
    LM_VALBOP_PATH = "/home/jure/datasets/LM/val_bop/lm_val_coco_gt.json"
    
    LMO_TRAIN_PATH = "/home/jure/datasets/LMO/synthetic/lmo_train_coco_gt.json"
    LMO_VAL_PATH = "/home/jure/datasets/LMO/val/lmo_val2_coco_gt.json"
    LMO_VALBOP_PATH = "/home/jure/datasets/LMO/val_bop/lmo_val_coco_gt.json"
    
    SYNTHETIC_TRAIN_PATH = "/home/jure/datasets/synthetic_dataset/synthetic_dataset_coco_gt.json"
        
    TRAIN_DATASETS = [AVD_TRAIN, GMU_TRAIN, TLESS_TRAIN, YCBV_TRAIN, SYNTHETIC_TRAIN_PATH] #TLESS_PATH, YCBV_PATH, , ICBIN_TRAIN_PATH, ICBIN_VAL_PATH
    TEST_DATASETS = [LM_VALBOP_PATH, LMO_VAL_PATH]

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
        
        if load_path is not None:
            with open(os.path.join(load_path, 'config.yml'), 'r') as yaml_file:
                config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for attr in list(config_dict.keys()):
                if not attr.startswith("__"):
                    setattr(Config, attr, config_dict[attr])
            print("Configuration is loaded: ")

        if save_path is not None:
            with open(os.path.join(save_path, 'config.yml'), 'w') as yaml_file:
                yaml.dump(config_dict, yaml_file, default_flow_style=False)

      