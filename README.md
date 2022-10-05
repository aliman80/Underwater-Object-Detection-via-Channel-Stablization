# Underwater-Object-Detection-via-Channel-Stablization
he complex marine environment exacerbates the
challenges of object detection manifold. With the advent of the
modern era, marine trash presents a danger to the aquatic
ecosystem, and it has always been challenging to address this
issue with complete grip. Therefore, there is a significant need
to precisely detect marine deposits and locate them accurately
in challenging aquatic surroundings. To ensure the safety of
the marine environment caused by waste, the deployment of
underwater object detection is a crucial tool to mitigate the
harm of such waste. Our work explains the image enhancement
strategies used and experiments exploring the best detection
obtained after applying these methods. Specifically, we evaluate
Detectron 2’s backbones performance using different base models
and configurations for the underwater detection task.
We first propose a channel stabilization technique on top of
a simplified image enhancement model to help reduce haze and
colour cast in training images. The proposed procedure shows
improved results on multi-scale size objects present in the data
set. After processing the images, we explore various backbones
in Detectron2 to give the best detection accuracy for these
images. In addition, we use a sharpening filter with augmentation
techniques. This highlights the profile of the object which helps us
recognize it easily. We demonstrate our results by verifying these
on TrashCan Data set, both instance and material version.
We then explore the best-performing backbone method for this
setting. We apply our channel stabilization and augmentation
methods to the best-performing technique. We also compare our
detection results from Detectron2 using the best backbones with
those from Deformable Transformer. The detection result for
small size objects in the Instance-version of TrashCan 1.0 gives us a
9.53\% absolute increase, in average precision while for the bounding
box we get the absolute gain of 7\% compared
to the baseline.\\  
