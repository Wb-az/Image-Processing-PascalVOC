import os
import xml.etree.ElementTree as ET

import numpy as np
from natsort import natsorted
import torch
from PIL import Image
from torch.utils import data as data

#####################################
# Class that takes as input bounding
# bboxes from  xml annotations and
# extracts binary masks
# This class was built based on
# https://learn-pytorch.oneoffcoder.com/object-detection.html
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#####################################


torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')


class PascalVoc(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = natsorted(os.listdir(os.path.join(root, 'Images')))
        self.annot = natsorted(os.listdir(os.path.join(root, 'annotations')))
        self.masks = natsorted(os.listdir(os.path.join(root, 'GT')))
        self.class_to_label = {name: i for i, name in enumerate(voc_classes)}
        self.idx_to_class = {i: name for i, name in enumerate(voc_classes)}
        assert len(self.annot) == len(self.imgs) == len(self.masks), 'Missing data'

    def __getitem__(self, idx):
        # load images and masks

        img_path = os.path.join(self.root, 'Images', self.imgs[idx])
        annot_path = os.path.join(self.root, 'annotations', self.annot[idx])
        mask_path = os.path.join(self.root, 'GT', self.masks[idx])
        # Convert image to RGB
        img = Image.open(img_path).convert('RGB')

        # get boxes coordinates and labels
        tree = ET.parse(annot_path)
        root = tree.getroot()

        labels = []
        bboxes = []
        iscrowd = []
        for obj in root.findall('object'):
            # get the class names
            class_name = obj.find('name').text
            difficult = int(obj.find('difficult').text)
            # bbox coordinates
            boxes = obj.find('bndbox')
            xmin = float(boxes.find('xmin').text) - 1
            ymin = float(boxes.find('ymin').text) - 1
            xmax = float(boxes.find('xmax').text) - 1
            ymax = float(boxes.find('ymax').text) - 1

            # store data in box coordinates and labels
            bboxes.append([xmin, ymin, xmax, ymax])
            class_name = self.class_to_label[class_name]
            labels.append(class_name)
            iscrowd.append(bool(difficult))

        mask = Image.open(os.path.join(mask_path))
        mask = np.array(mask)
        # loop through all boxes to get the masks  
        masks = []
        for i, box in enumerate(bboxes):
            msk_array = np.zeros_like(mask, dtype=np.uint8)

            # crop the bounding box from the image mask
            box = [int(x) for x in box]
            crop = mask[box[1]:box[3], box[0]:box[2]]
            msk_array[box[1]:box[3], box[0]:box[2]] = crop

            # Convert the mask to binary masks, matches set to 1
            msk_array[msk_array == labels[i]] = 1
            msk_array[msk_array == 255] = 1
            msk_array[msk_array != 1] = 0
            masks.append(msk_array)

        # Convert to tensors
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # iscrowd = torch.zeros((len(bboxes)), dtype=torch.int64)
        iscrowd = torch.BoolTensor(iscrowd)

        # area - used to separate the metric scores among small, medium and large boxes
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        assert len(bboxes) == len(labels) and len(bboxes) == len(masks), 'Reprocess annotations'

        target = {'boxes': bboxes, 'labels': labels, 'masks': masks, 'image_id': image_id,
                  'area': area, 'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]
