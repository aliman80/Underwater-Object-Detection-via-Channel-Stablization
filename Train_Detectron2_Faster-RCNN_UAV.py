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
                        "/home/muhammad.ali/Desktop/Research/Wast_net/TACO/data/annotations/annotations_0_train.json",
                        "/home/muhammad.ali/Desktop/Research/Wast_net/TACO/data/images/")
register_coco_instances("data_val", {}, 
                        "/home/muhammad.ali/Desktop/Research/Wast_net/TACO/data/annotations/annotations_0_val.json",
                        "/home/muhammad.ali/Desktop/Research/Wast_net/TACO/data/images/")


# ## Train!
# 
# Now, let's fine-tune a COCO-pretrained R101-FPN Mask R-CNN model on the iSAID dataset.

# In[4]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# cfg.MODEL.RETINANET.NUM_CLASSES = 16
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.7
#cfg.OUTPUT_DIR = '/home/muhammad.ali/Downloads/RESULTSFOLDER/Enhanced/FasterRCNN/Material/feb15lr0.0025r101'
cfg.OUTPUT_DIR = '/home/muhammad.ali/Downloads/detec/RESULTSFOLDER/Without Enhanced/TACO_Result'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("data_train",)
# cfg.INPUT.RANDOM_FLIP = "horizontal"
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025 # pick a good LR # orignal LR WAS 0.0005 0.0025
#cfg.build_lr_scheduler(cfg.SOLVER.BASE_LR) 
# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
# cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.MAX_ITER = 40000 
# cfg.SOLVER.STEPS = [20000, 30000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 67
# cfg.MODEL.RPN.NMS_THRESH = 0.7


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



