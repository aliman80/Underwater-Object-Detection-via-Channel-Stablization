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
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader


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
                        "/home/muhammad.ali/Desktop/dataset/instance_version/instances_train_trashcan.json",
                        "/home/muhammad.ali/Desktop/dataset/instance_version/train")
register_coco_instances("data_val", {}, 
                        "/home/muhammad.ali/Desktop/dataset/instance__version/instances_val_trashcan.json",
                        "/home/muhammad.ali/Desktop/dataset/instance__version/val")


# ## Train!
# 
# Now, let's fine-tune a COCO-pretrained R101-FPN Mask R-CNN model on the iSAID dataset.

# In[4]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.RETINANET.NUM_CLASSES = 22
cfg.OUTPUT_DIR = 'output_Reinanet'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("data_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025 # pick a good LR # orignal LR WAS 0.0005
#cfg.build_lr_scheduler(cfg.SOLVER.BASE_LR) 
cfg.SOLVER.MAX_ITER = 40000 
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 22

##### CustomModule ####### CustomModule
class Trainer(DefaultTrainer):
    def build_sem_seg_train_aug(cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        augs.append(T.RandomFlip())
        return augs
    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)







os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

####### CustomModule

dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((800, 800))
   ]))
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()
trainer = Trainer(DefaultTrainer)
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
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "output_Reinanet/model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# In[6]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("data_val")
val_loader = build_detection_test_loader(cfg, "data_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

# another equivalent way to evaluate the model is to use `trainer.test`






  
# '''
# Author:Xuelei Chen(chenxuelei@hotmail.com)
# '''
# import torch
# import numpy as np
# import torch.nn as nn


# class AConvBlock(nn.Module):
#     def __init__(self):
#         super(AConvBlock,self).__init__()

#         block = [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.AdaptiveAvgPool2d((1,1))]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()]
#         self.block = nn.Sequential(*block)

#     def forward(self,x):
#         return self.block(x)

# class tConvBlock(nn.Module):
#     def __init__(self):
#         super(tConvBlock,self).__init__()

#         block = [nn.Conv2d(6,8,3,padding=1,dilation=1)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(8,8,3,padding=2,dilation=2)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(8,8,3,padding=5,dilation=5)]
#         block += [nn.PReLU()]

#         block += [nn.Conv2d(8,3,3,padding=1)]
#         block += [nn.PReLU()]
#         self.block = nn.Sequential(*block)
#     def forward(self,x):
#         return self.block(x)

# class PhysicalNN(nn.Module):
#     def __init__(self):
#         super(PhysicalNN,self).__init__()

#         self.ANet = AConvBlock()
#         self.tNet = tConvBlock()

#     def forward(self,x):
#         A = self.ANet(x)
#         t = self.tNet(torch.cat((x*0+A,x),1))
#         out = ((x-A)*t + A)
#         return torch.clamp(out,0.,1.)