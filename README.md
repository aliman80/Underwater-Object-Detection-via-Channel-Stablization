Underwater-Object-Detection-via-Channel-Stablization
The complex marine environment exacerbates the hallenges of object detection manifold. With the advent of them modern era, marine trash presents a danger to the aquaticecosystem, and it has always been challenging to address thisi ssue with complete grip. Therefore, there is a significant needt to precisely detect marine deposits and locate them accuratelyin challenging aquatic surroundings. To ensure the safety ofthe marine environment caused by waste, the deploy-ent of uderwater object detection is a crucial tool to mitigate theharm of such waste. Our work explains the image enhancement strategies used experiments exploring the best detectiono btained after applying these methods. Specifically, we evaluateDetectron 2’s backbones performance using different base models and configurations for the underwater detection task.We first propose a channel stabilization technique on top of a simplified image enhancement model  elp reduce haze and colour cast in training images. The proposed procedure showsi mproved results on multi-scale size objects present in the dataset. After processing the images, we explore various backbonesin Detectron2 to give the best detection accuracy for theseimages. In addition, we use a sharpening filter with augmentationt echniques. This highlights the profile of the object which helps usr ecognize it easily. We demonstrate our results by verifying theseoin Tthe rashCan Data set, both instance and material version. We then explore the best-performing backbone method for this setting. We apply our channel stabilization and augmentation methods to the best-performing technique. We also compare oureresults from Detectron2 using the best backbones with those from Deformable Transformer. The detection result  for small size objects in the Instance-version of TrashCan 1.0 gives us a 9 .53\% absolute increase, in average precision  At the same time,for the boundingb ox we get the absolute gain of 7\% comparedt o the baseline. 
## Using simplified RGHS method
<img width="243" alt="Screen Shot 2023-02-07 at 7 06 10 PM" src="https://user-images.githubusercontent.com/57188476/217282454-064a850b-547f-472f-a6be-3705045f3f07.png">
Preprocessing of the given input image using Relative global historgram model. 

## Channel Stablizaition method
Application of Channel stablzation method to reduce the dominance of blue colour under deep water to classify and detect waste better.
<img width="485" alt="Screen Shot 2023-02-07 at 7 08 46 PM" src="https://user-images.githubusercontent.com/57188476/217283088-68826e81-27fc-425d-a3bb-755c44d71353.png">



## Using Detectron2 for Comparison
Then we use Detectron2 for our detection , we use different backbones including RetinaNet and FasterRCNN.
<img width="882" alt="Screen Shot 2023-02-07 at 6 48 46 PM" src="https://user-images.githubusercontent.com/57188476/217277849-cc3d6a19-75ad-4310-8ae2-73356e066502.png">
