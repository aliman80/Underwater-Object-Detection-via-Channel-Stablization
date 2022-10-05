#!/usr/bin/env python
# coding: utf-8

# # Tutorial adapted from the Detectron2 colab example

# # Install detectron2
# https://github.com/facebookresearch/detectron2
# 
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# In[2]:


# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())


# In[2]:


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# # Train on a iSAID dataset

# In this section, we show how to train an existing detectron2 model on the iSAID dataset.
# 
# 
# ## Prepare the dataset

# Since iSAID is in COCO format, it can be easily registered in Detectron2

# In[3]:



# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.data.datasets import register_coco_instances
register_coco_instances("data_train", {}, 
                        "/home/muhammad.ali/Downloads/RESULTSFOLDER/dataset/material_version/instances_train_trashcan.json",
                        "/home/muhammad.ali/Downloads/detec/RESULTSFOLDER/dataset/material_version/train")
register_coco_instances("data_val", {}, 
                        "/home/muhammad.ali/Downloads/RESULTSFOLDER/dataset/material_version/instances_val_trashcan.json",
                        "/home/muhammad.ali/Downloads/detec/RESULTSFOLDER/dataset/material_version/val")


# ## Train!
# 
# Now, let's fine-tune a COCO-pretrained R101-FPN Mask R-CNN model on the iSAID dataset.

# In[4]:


from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultTrainer
cfg = CN()
cfg = get_cfg()
cfg.model = CN()
cfg.MODEL.RETINANET.NUM_CLASSES = 16
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = '/home/muhammad.ali/Downloads/detec/RESULTSFOLDER/Enhanced/Retina/material/march3 '
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("data_train",)
# cfg.INPUT.RANDOM_FLIP = "horizontal"
# cfg.INPUT.CROP = CN({"ENABLED": True})
# cfg.INPUT.CROP.TYPE = "relative_range"
# cfg.INPUT.CROP.SIZE = [0.9, 0.9]
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 # pick a good LR # orignal LR WAS 0.0005
#cfg.build_lr_scheduler(cfg.SOLVER.BASE_LR) 
# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
# cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.MAX_ITER = 40000 
# cfg.SOLVER.STEPS = [20000, 30000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
cfg.MODEL.RPN.NMS_THRESH = 0.5


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# ### Look at training curves in tensorboard by running in the terminal:
# tensorboard --logdir output_fasterrcnn

# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the validation dataset. First, let's create a predictor using the model we just trained:
# 
# 

# In[5]:


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# In[6]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("data_val")
val_loader = build_detection_test_loader(cfg, "data_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`



