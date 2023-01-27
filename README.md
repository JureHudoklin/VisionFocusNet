# VisionFocusNet

<!-- q: how to insert image into md file -->
![alt text](/example/test_output/test_2.jpg_valve.png "Example Output")
**Template based object detection.**


## Requirements and Installation

This code is tested on Ubuntu 20.04 with Python 3.8.5 and PyTorch 1.10.1.
We recommend using a virtual environment.

```bash
python3 -m venv env
source env/bin/activate
```

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

You also need to install the Deformable Transformer CUDA packages. To do so navigate to "models/ops" and run the following commands:

```bash
./make.sh
```

## Usage
To use the network you can either use the pretrained weights we provide or train the network yourself.

### Pretrained Models
The pretrained model weights may be downloaded from the following link:

https://something.com


### Training from Scratch
To train the network you first need to setup your dataset.
The dataset annotation file is a modified version of a COCO dataset.
The dataset should have the following folder structure:

```bash
my_dataset/
├── images/
|   ├── img_1.{ext}
|   ├── img_2.{ext}
├── depth/
|   ├── img_1.{ext}
|   ├── img_2.{ext}
├── targets/
|   ├── {tgt_class_1}
|   |    ├── tgt_img_1.{ext}
|   ├── {tgt_class_2}
|   |    ├── tgt_img_1.{ext}
annotations.json
```
        
Annotations.json is a modified version of COCO annotations.
    It has the following structure (Filename should include the extension):

```python
{
    "images": [ {"id": int, "file_name": str, "depth_name": str, "width": int, "height": int}, ... ],
    "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h], "area": int, "iscrowd": float, "sup_id": int}, ... ],
    "categories": [{"id": int, "name": str, "supercategory": str, "sup_id": int}, ... ],
}
```

Next set up your datasets in the config file (configs/visual_focusnet_config.py). You can set the "TRAIN_DATASETS" and "VAL_DATASETS" variables. Each variable should contain paths to your annotation.json files.

In the config file you can also set other network parameters.

To train the network run the following command:

```bash
python train_detr.py --save_dir <path_to_save_dir> --load_dir <OPTIONAL: path_to_load_dir> 
```

### Testing
To test the network on your data you can use "test_detr.py" file.
First prepare your data as shown in the "**example/data**" folder.
Next run the testing:

```bash
python test_detr.py --load_dir <path_to_load_dir> --save_dir <path_to_save_dir> --data_dir <path_to_data_dir>
```

## Citation
If you use this code for your research, please cite my thesis:

** Suction Grasp Generation and Template-Based object detection **

Some parts of the code are based on the following repositories:
- [DETR](https://github.com/facebookresearch/detr)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [DINO-DETR](https://github.com/IDEA-Research/DINO)


