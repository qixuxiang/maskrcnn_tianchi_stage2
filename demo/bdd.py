# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
import time
from collections import defaultdict
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

class BDD:        
  def __init__(self, annotation_file=None):
    """
    Constructor of Microsoft BDD helper class for reading and visualizing annotations.
    :param annotation_file (str): location of annotation file
    :param image_folder (str): location to the folder that hosts images.
    :return:
    """
    # load dataset
    self.cats =  []
    self.cats_dict = {}
    self.dataset,self.imgs,self.imgs_info = list(),list(), list()
    self.attributes,self.labels,self.bboxes = dict(),dict(),dict()
    self.imgToLabs, self.catToImgs = defaultdict(list), defaultdict(list)
    if not annotation_file == None:
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        # print(dataset['info'])
        # print(dataset['licenses'])
        # print(dataset['categories'])
        # print(type(dataset['images']))
        # print(len(dataset['images']))
        # print((dataset['images'][0]))
        # print((dataset['images'][1]))
        # print((dataset['images'][2]))
        # print(type(dataset['annotations']))
        # print(len(dataset['annotations']))
        # print(dataset['annotations'][0])
        # print(dataset['annotations'][1])
        # print(dataset['annotations'][2])
        # print(dataset['annotations'][3])

        # assert type(dataset)==list, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()
            
  def createIndex(self):
      # create index
      print('creating index...')
      self.cats_dict = self.dataset['categories']
      # print(self.cats_dict)
      for cat in self.cats_dict:
          # print(cat)
          self.cats.append(cat['name'])
      # print(self.cats)
      
      for img_info in self.dataset['images']:
          # print(img_info['file_name'],"   ", img_info['height'],"  ", img_info['width'])
          img_info_dict = {'id':img_info['id'], 'file_name': img_info['file_name'], 'height': img_info['height'], 'width': img_info['width'] }
          self.imgs_info.append(img_info_dict)
          img = img_info['file_name'][:-4:]
          self.imgs.append(img)
      # print(img)
      # print(len(self.imgs))
      # print(len(self.imgs_info))

      bboxes = {}
      boxes = list()

      i = 0
      anno_len = len(self.dataset['annotations']) 
      # print(anno_len)
      for img_info in self.imgs_info:
          if not self.dataset['annotations']:
            break
          annotation = self.dataset['annotations'][i]
          img = img_info['id']
          # height = img_info['height']
          # width  = img_info['width']
          while(annotation['image_id'] == img):
              xmin = annotation['bbox'][0]
              ymin = annotation['bbox'][1]
              xmax = annotation['bbox'][0] + annotation['bbox'][2]
              ymax = annotation['bbox'][1] + annotation['bbox'][3]
              # print(xmin)
              if (xmax > xmin and ymax > ymin):
                  box = {'category_id': annotation['category_id'], 'bbox': [xmin, ymin, xmax, ymax]}
                  boxes.append(box)
              i += 1
              if (i < anno_len):
                  annotation = self.dataset['annotations'][i]
              else:
                  break
              
          temp_boxes = boxes.copy()
          bboxes[img] = temp_boxes
          boxes.clear()
      
      # print(len(bboxes))
      # for img, bbox in bboxes.items():
      #     # print(img)
      #     if (len(bbox) == 0):
      #         print(img)
          # print(len(bbox)) 
      
      
      # # create class members
      # self.imgs = imgs
      # self.attributes = attrs
      # self.labels = labs
      self.bboxes = bboxes
      # print('-------------------------------------')
      # print(len(self.bboxes))
        

  def loadCats(self):
      """
      Load cats with the specified ids.
      :return: cats (object array) : loaded cat objects
      """
      return self.cats

  def getImgIds(self):
      """
      Load cats with the specified ids.
      :return: imgs (object array) : loaded cat objects
      """
      return self.imgs

  def getImgHW(self,index):
      """
      Load cats with the specified ids.
      :return: imgs (object array) : loaded cat objects
      """
      height = self.imgs_info[index]['height']
      width = self.imgs_info[index]['width']
      return height, width

  def loadBboxes(self, index):
      """
      Load cats with the specified ids.
      :return: bbox (object array) : loaded cat objects
      """
      # print(self.bboxes.get(index))
      return self.bboxes.get(index)

  # _BDD.loadBboxes(index)
  
  def loadAttributes(self, index):
      """
      Load cats with the specified ids.
      :return: bbox (object array) : loaded cat objects
      """
      # print(self.bboxes.get(index))
      return self.attributes.get(index)

class bdd(imdb):
  def __init__(self, image_set, year):
    imdb.__init__(self, 'bdd_' + year + '_' + image_set)
    # COCO specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    
    self.widths = 1280
    self.heights = 720
    # name, paths
    self._year = year
    self._image_set = image_set
    # self._data_path = osp.join(cfg.DATA_DIR, 'bdd100k') 
    self._data_path = osp.join(cfg.DATA_DIR, 'jinnan') 
    
    # load COCO API, classes, class <-> id mappings
    self._BDD = BDD(self._get_ann_file())
    cats = self._BDD.loadCats()
    # print(cats)
    self._classes = ['__background__'] + cats
    # print((self.classes))

    num_classes = len(self._classes)
    self._class_to_ind = dict(zip(self.classes, range(num_classes)))
    self._ind_to_class = dict(zip(range(num_classes), self._classes))
    self._image_index = self._load_image_set_index()
    print('---------------------image_index-----------------------')
    # print((self._image_index))

    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # For example, minival2014 is a random 5000 image subset of val2014.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'train2018': 'train',  # 5k val2014 subset
      'val2018': 'val',
      'test2018': 'test'
    }
    bdd_name = image_set + year  # e.g., "val2014"
    self._data_name = (self._view_map[bdd_name]
                       if bdd_name in self._view_map
                       else bdd_name)
    # print('----------------------------data_name-----------------------------')
    # print(self._data_name)
    # # Dataset splits that have ground-truth annotations (test splits
    # # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'test')

  def _get_ann_file(self):
    prefix = self._image_set+'_no_poly' #if self._image_set.find('test') == -1 \
    #  else 'image_info'
    return osp.join(self._data_path, 'labels',prefix +'.json')

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._BDD.getImgIds()
    return image_ids

  def _get_widths(self):
    # anns = self._COCO.loadImgs(self._image_index)
    # widths = [ann['width'] for ann in anns]
    return self.widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    # print('----------image_index---------------')
    # print(self.image_index[i])
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = (index + '.jpg')
    if self._image_set == 'train':
      file_path = 'jinnan2_round1_train_20190305/restricted'
    else:
      file_path = 'jinnan2_round1_test_a_20190306'
    image_path = osp.join(self._data_path, file_path, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    # if osp.exists(cache_file):
    #   with open(cache_file, 'rb') as fid:
    #     roidb = pickle.load(fid)
    #   print('{} gt roidb loaded from {}'.format(self.name, cache_file))
    #   return roidb
    
    gt_roidb = [self._load_bdd_annotation(index)
                for index in range(0, len(self._image_index))]
    
    # gt_roidb = [self._load_bdd_annotation(0)]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_bdd_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    # width = self.widths
    # height = self.heights
    # print(index)
    objs = self._BDD.loadBboxes(index)  
    height, width = self._BDD.getImgHW(index)  
    # print(objs)
    # print(height)
    # print(width)

    valid_objs = []
    if self._image_set == 'train':
      for obj in objs:
        x1 = np.max((0, obj['bbox'][0]))
        y1 = np.max((0, obj['bbox'][1]))
        x2 = np.max((0, obj['bbox'][2])) 
        y2 = np.max((0, obj['bbox'][3])) 
        if x2 >= x1 and y2 >= y1:
          obj['bbox'] = [x1, y1, x2, y2]
          valid_objs.append(obj)
    
    objs = valid_objs
    num_objs = len(objs)    
    # num_objs = 0   

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class
    
    if self._image_set == 'train':
      for ix, obj in enumerate(objs):
        cls = obj['category_id']
        boxes[ix, :] = obj['bbox']
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        x1 = obj['bbox'][0]
        y1 = obj['bbox'][1]
        x2 = obj['bbox'][2]
        y2 = obj['bbox'][3]
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
            }

  # def _get_widths(self):
  #   return self.widths

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    print('--------------num_images---------')
    print(num_images)
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      # print(boxes)
      # oldx1 = boxes[:, 0].copy()
      # oldx2 = boxes[:, 2].copy()
      # boxes[:, 0] = widths - oldx2 - 1
      # boxes[:, 2] = widths - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths,
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      # break
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = ('COCO_' + self._data_name +
                 '_' + str(index).zfill(12) + '.mat')
    return osp.join(file_name[:14], file_name[:22], file_name)

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


  def _do_detection_eval(self, res_file, output_dir):
    # print('-------------------------')
    # print(output_dir)
    # print(res_file)
    # print(output_dir)
    gt = 'eval/vgg16/bdd/bdd_val.json'
    result = res_file
    # mean, breakdown = evaluate_detection(gt, result)
    # print('{:.2f}'.format(mean),
      # ', '.join(['{:.2f}'.format(n) for n in breakdown]))

  def _bdd_results_one_category(self, boxes, cat):
    results = []
    # i = 0 
    for im_ind, index in enumerate(self._image_index):
            # i = i + 1 
            # if(i == 40):
            #     break
                
      # print('im_ind: ', im_ind)
      # print('index: ', index)
      img_name = index + '.jpg'
      dets = boxes[im_ind].astype(np.float)
      # print(dets)
      if dets == []:
        continue
      scores = dets[:, -1]
      x1s = dets[:, 0]
      y1s = dets[:, 1]
      ws = dets[:, 2] - x1s + 1
      hs = dets[:, 3] - y1s + 1
      x2s = x1s + ws
      y2s = y1s + hs

      results.extend(
        [{'filename' : img_name,
          'rects':[{"xmin": int(x1s[k]),"xmax": int(x2s[k]), "ymin": int(y1s[k]), "ymax": int(y2s[k]), "label": cat,"confidence": round(scores[k], 3)}]
          } for k in range(dets.shape[0])])

            # break
        # print(results)
    return results

  def _write_bdd_results_file(self, all_boxes, res_file):
        # [{"name": str,
        #   "timestamp": 1000,    
        #   "category": str,
        #   "bbox": [x1, y1, x2, y2],
        #   "score": 0.236}]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                          self.num_classes ))
            # print('------------------------------------------------')
            # print(cls_ind, ' ', cls)
            # bdd_cat =  self._ind_to_class[cls_ind] 

            # print(bdd_cat)
            results.extend(self._bdd_results_one_category(all_boxes[cls_ind],
                                                           cls_ind))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
            # break
        #res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid, indent=4, separators=(',', ': '))

  def evaluate_detections(self, all_boxes, output_dir, checkepoch):
        res_file = os.path.join(output_dir, ('bdd_' +
                                         self._image_set +
                                         '_results_'+ checkepoch))
        res_file += '.json'
        # print('-------------all_bxoes-------------')
        # # print(all_boxes.size())
        self._write_bdd_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file  
