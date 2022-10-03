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
    DECAY = 5e-6  # I Change it from 1e-4
    BATCH_SIZE = 2
    EPOCH = 20
    SAVE_BEST_ONLY = False
    DROPOUT_RATE = 0.1

    #########
    # Model #
    #########
    IMG_HEIGHT = 270 #540
    IMG_WIDTH = 480 #960
    TARGET_HEIGHT = 32
    TARGET_WIDTH = 32
    
    NUM_OF_QUERIES = 100
    TRANSFORMER_INPUT_SIZE = 448
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6

    ##############

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

        if save_path is not None:
            with open(os.path.join(save_path, 'config.yml'), 'w') as yaml_file:
                yaml.dump(config_dict, yaml_file, default_flow_style=False)

        if load_path is not None:
            with open(os.path.join(load_path, 'config.yml'), 'r') as yaml_file:
                config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for attr in list(config_dict.keys()):
                if not attr.startswith("__"):
                    setattr(Config, attr, config_dict[attr])
            print("Configuration is loaded: ")
            print(config_dict)