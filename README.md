<h1><div style="text-align: center;"> COVID-19 Computer Vision: Pascal VOC dataset</div></h1>

<h2><div style="text-align: center;"> Instance Segmentation and detection of Pascal VOC objects </div></h2>

<h3><div style="text-align: center;"> A Ascencio-Cabral
</div></h3>


## Models

- Faster-RCNN-50-FPN -pretrained off-the-box pytorch network
- Mask-RCNN-50-FPN - pretrained off-the-box  pytorch network
- Mask-RCNN-101-FPN - built with pretrained backbone on ImageNet dataset
- Mask-RCNN-101-FPN with customised anchors sizes=(16, 32, 64, 128, 256, 512) - built with pretrained backbone on ImageNet dataset

### Evaluation - Coco style metrics
 - Mean Average Precision (AP or mAP) at IoU [0.5, 0.05, 0.95], 0.75 and 0.50


## 1. Introduction

With the developments in deep learning, the applicability of computer vision has been widely spread in fields such as robotics, image search, recognition and autonomous driving. In this work Mask-RCNN-ResNet50-FPN, Mask-RCNN-ResNet101-FPN, Mask-RCNN-ResNet101-FPN with customised anchor sizes and Faster-ResNet50-FPN were used for the instance semantic segmentation and object detection on the PASCAL VOC 2012 dataset.


## 3. Methods

### 3.1 Environment

A python environment was setup and the experiments were built using Pytorch. 

### 3.2 Datasets
For this project only the images, annotations and segmented class masks of the Pascal VOC 2012 kit dataset were used [1]. This dataset has 21 classes including the background. The dataset contained in total 2913 images with annotations and ground truth. The holdout method with proportion of 80:10:10 for training, validation and test of the models.

**voc_classes**: ```' __background__ ','aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus','car','cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                           'tvmonitor'```

### 3.3 Training

The approach to training was transfer learning and fine tuning the last three layers of the models. The models were trained to minimize the loss with SGD and Adam optimizers, a weght decay of 0.0001 and learning rates of 0.001, 0.005 and 0.0001 with a step scheduler for 5 steps and gamma of 0.2. All models were trained on google colab.


### 3.4 Evaluation metrics

The performance of the model was measured after each epoch training on the validation subset and on the test subset after training. The coco style mean average precision was  measure at IoU thresholds [0:50,0:05,0:95], 0.50 and 0.75 [2]. The mean average precision (mAP) was computed for all classes and per each class on the test subset [3].


## 4. Results - Coco style metrics

Tables 1-2 show the mean average precision of the best models for all classes and per class, respectively. F

**Table 1.** Best models performance on the test dataset. The mean average precision is given in percentage for all Pascal VOC classes 2012. All results are shown in percentage. Mask-RCNN-ResNet101-FPN-CA has customised anchors sizes=(16, 32, 64, 128, 256, 512)


|  Network                  | Epochs |   lr  | Optimizer  | mAP @IoU </br>[0:50,0:05,0:95] </br> detection | mAP @IoU=50 </br> detection |mAP @IoU=75 </br>detection | mAP @IoU </br>[0:50,0:05,0:95] </br> segmentation| mAP @IoU=50 </br> segmentation|mAP @IoU=75 </br> segmentation |
|:-------------------------:|:------:|-------:|:----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Faster-RCNN-ResNet50-FPN  |  15    | 0.005  |   SGD      |   89.2   |  99.00   |   97.5   |  NA.     |   NA     |   NA     | 
| Mask-RCNN-ResNet50-FPN    |  15    | 0.005  |   SGD      |   54.0   |  82.30   |   60.8   |  43.6    |  71.0    |  45.7    |  
| Mask-RCNN-ResNet101-FPN   |  20    | 0.0001 |   Adam     |   42.1   |  72.2   |   44.4  |  38.2   | 64.2    |  39.8    | 
| Mask-RCNN-ResNet101-FPN-CA |  20    | 0.0001 |   Adam     |   43.7   |  69.9   |   48.8   |  39.2   |   63.9   |   42.5  |




**Table 2.** Best models performance on the test dataset. The mean average precision is given per each object class of the Pascal VOC 2012 dataset. All results are shown in percentage. Mask-RCNN-ResNet101-FPN-CA has customised anchors sizes=(16, 32, 64, 128, 256, 512).

|Model|Task  |aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Mask-RCNN-ResNet101-FPN |detection|87\.75|73\.62|88\.98|61\.21|60\.6|76\.39|82\.27|82\.16|43\.62|67\.73|52\.76|79\.74|73\.55|84\.82|83\.57|45\.36|68\.53|67\.94|
|Mask-RCNN-ResNet101-FPN |segmentation|87\.75|0\.0|87\.92|45\.19|58\.74|74\.97|79\.5|81\.36|22\.66|58\.39|55\.85|79\.74|76\.69|68\.89|81\.59|31\.49|68\.53|68\.4|
|Mask-RCNN-ResNet101-FPN-CA|detection|90\.33|66\.14|86\.56|56\.24|46\.77|79\.13|86\.05|84\.19|39\.65|66\.21|56\.94|79\.9|72\.91|84\.65|85\.16|42\.47|58\.95|68\.2|
|Mask-RCNN-ResNet101-FPN-CA|segmentation|88\.12|0\.0|84\.63|45\.96|46\.77|80\.85|77\.05|75\.22|30\.09|72\.22|63\.64|76\.52|76\.81|74\.83|80\.62|26\.84|64\.42|62\.88|

## References

[1] The PASCAL Visual Object Classes Homepage’. http://host.robots.ox.ac.uk/pascal/VOC/

[2] COCO - Common Objects in Context’. https://cocodataset.org/#detection-eval (accessedAug. 02, 2020).

[3] K. Morabia, J. Arora, and T. Vijaykumar, ‘Attention-based Joint Detection of Object and Semantic Part’, arXiv:2007.02419 [cs], Jul. 2020, Accessed: Jul. 02, 2020. [Online]. Available: http://arxiv.org/abs/2007.02419


