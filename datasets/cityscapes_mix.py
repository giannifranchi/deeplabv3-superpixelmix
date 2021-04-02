import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

from skimage.morphology import erosion, dilation,binary_erosion, opening, closing, white_tophat, reconstruction, area_opening
from skimage.morphology import black_tophat, skeletonize, convex_hull_image,extrema
from skimage.morphology import square, diamond, octagon, rectangle, star, disk, label
from skimage.segmentation import watershed


class Cityscapes_mix(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None,watershed=False,watercutout=0.3,nb_markers=200):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        self.watercutout=watercutout
        self.nb_markers = nb_markers
        self.watershed = watershed
        if self.watershed: self.watershed_mask = Cutoutwatershed_cityscape(self.watercutout, self.nb_markers)

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        if self.watershed: mask = self.watershed_mask(image)
        target = self.encode_target(target)
        if self.watershed : return image, target, mask
        else:  return image, target

        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


se1_0 = np.array([[ 0, 1, 0],
              [ 0, 1, 0],
              [ 0, 1, 0]], dtype=np.uint8)


se2_0 = np.array([[ 0, 0, 0],
              [ 1, 1, 1],
              [0, 0, 0]], dtype=np.uint8)


se3_0 = np.array([[ 0, 0, 1],
              [0,  1, 0],
              [1, 0, 0]], dtype=np.uint8)


se4_0 = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=np.uint8)




se1_1 = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]], dtype=np.uint8)


se2_1 = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.uint8)


se3_1 = np.array([[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]], dtype=np.uint8)


se4_1 = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]], dtype=np.uint8)



se1_2 = np.array([[0,0,0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)


se2_2 = np.array([[0,0,0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0,0, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)


se3_2 = np.array([[0,0,0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

se4_2 = np.array([[1,0,0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)

scale0=[se1_0,se2_0,se3_0,se4_0]
scale1=[se1_1,se2_1,se3_1,se4_1]
scale2=[se1_2,se2_2,se3_2,se4_2]


def gradient(rgb):
    # description
    # input :image rgb
    # output : contour
    rgb = np.asarray(rgb).astype(np.uint8)
    lab = color.rgb2lab(rgb)

    '''print(np.amin(lab[:, :, 0]), np.amax(lab[:, :, 0]))
    print(np.amin(lab[:, :, 1]), np.amax(lab[:, :, 1]))
    print(np.amin(lab[:, :, 2]), np.amax(lab[:, :, 2]))'''
    ##Sobel operator kernels.
    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale0)):
        imgGrad = dilation(lab[:,:,0], scale0[i]) - erosion(lab[:,:,0], scale0[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale0[i]) - erosion(lab[:,:,1], scale0[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale0[i]) - erosion(lab[:,:,2], scale0[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale0=np.maximum(grad_L, grad_a,grad_b)

    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale1)):
        imgGrad = dilation(lab[:,:,0], scale1[i]) - erosion(lab[:,:,0], scale1[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale1[i]) - erosion(lab[:,:,1], scale1[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale1[i]) - erosion(lab[:,:,2], scale1[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale1=np.maximum(grad_L, grad_a,grad_b)

    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale2)):
        imgGrad = dilation(lab[:,:,0], scale2[i]) - erosion(lab[:,:,0], scale2[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale2[i]) - erosion(lab[:,:,1], scale2[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale2[i]) - erosion(lab[:,:,2], scale2[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale2=np.maximum(grad_L, grad_a,grad_b)

    grad=(grad_scale0+grad_scale1+grad_scale2)/3

    return grad

def mosaic_cutout_binary(rgb,labels_waterhed,prop):
    shape = np.shape(rgb)
    mask = np.ones((shape[0],shape[1])).astype(np.float16)
    '''r,g,b = np.split(mask, 3, axis=2)
    mask_r = r
    mask_g = g
    mask_b = b'''
    nb_cluster=np.amax(labels_waterhed)
    perm = np.random.permutation(nb_cluster)
    nb=int(nb_cluster*prop)
    nb_cluster=perm[0:nb]
    for i in nb_cluster:
        mask[labels_waterhed == i] = 0

    return mask

class Cutoutwatershed_cityscape(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, prop,nb_markers=200):
        self.prop = prop
        self.nb_markers=nb_markers

    def __call__(self, img):
        if self.prop <= 0:
            return img

        #img2=Image.fromarray(img2)
        print(img)
        print(img.min(),img.max())

        size=img.size()
        img2 = np.ones((size[1],size[2],3)).astype(np.uint8)
        r,g,b = np.split(img.numpy(), 3, axis=0)
        img2[:, :, 0] = r
        img2[:, :, 1] = g
        img2[:, :, 2] = b

        img2=Image.fromarray(img2)

        grad =gradient(img2)
        labels_waterhed = watershed(grad, markers=200, compactness=0.001)
        img=mosaic_cutout_binary(img2,labels_waterhed,self.prop)
        #Image.fromarray(segments_watershed).show('img_mosaic')
        #print(img)

        return img
