import os

import ipdb
import matplotlib
from tqdm import tqdm
import torch as t

import numpy as np
from scipy.misc import imsave
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from data.util import read_image
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def test_max(**kwargs):
    
    opt._parse(kwargs)
    file_path='/home/huachunrui/model_h5/img/test'
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")
    trainer.load('/home/huachunrui/model_h5/checkpoints/fasterrcnn_09071338_0.7790970180127188')
    file_list=os.listdir(file_path)
    for i,name in enumerate(file_list):
        # if i>=20:
        path=os.path.join(file_path+'/',name)
        print(path)
        img=read_image(path)
        img=t.from_numpy(img)[None]

        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, visualize=True)
        #print(pred_scores_[0].reshape(-1))
        #re_pred_scores_=[i for i in pred_scores_[0] if i>=0.75]
        index=[i for i in range(len(pred_scores_[0])) if pred_scores_[0][i]>=0.75]
        # print(pred_bboxes_)
        # print(pred_scores_)

        # print(type(pred_bboxes_))

        # print(index)
        # print('**********************')

        # print(pred_bboxes_[0])
        # print(type(pred_bboxes_[0]))
        # print(pred_scores_[0])
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^')

        # print(len(pred_bboxes_[0][index]))
        # print(len(pred_scores_[0][index]))
        # print(pred_scores_[0][index].reshape(-1))


        pred_img = visdom_bbox(at.tonumpy(img[0]),
                            at.tonumpy(pred_bboxes_[0][index]),
                            at.tonumpy(pred_labels_[0][index]).reshape(-1),
                            at.tonumpy(pred_scores_[0][index]))
        imsave('/home/huachunrui/model_h5/img/{}'.format(name),(255*pred_img).transpose(1,2,0))

        # if i==30:
        #     break

def test_all(**kwargs):
    
    opt._parse(kwargs)
    file_path='/home/huachunrui/data_h5/imgs'
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")
    trainer.load('/home/huachunrui/model_h5/checkpoints/fasterrcnn_09071338_0.7790970180127188')
    file_list=os.listdir(file_path)
    for i,name in enumerate(file_list):
        if i==10:break
        path=os.path.join(file_path+'/',name)
        print(path)
        img=read_image(path)
        img=t.from_numpy(img)[None]

        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, visualize=True)
        #print(pred_scores_[0].reshape(-1))
        #re_pred_scores_=[i for i in pred_scores_[0] if i>=0.75]
        #print(re_pred_scores_)


        pred_img = visdom_bbox(at.tonumpy(img[0]),
                           at.tonumpy(pred_bboxes_[0]),
                           at.tonumpy(pred_labels_[0]).reshape(-1),
                           at.tonumpy(pred_scores_[0].reshape(-1)))
        imsave('/home/huachunrui/model_h5/img/all_2/{}'.format(name),(255*pred_img).transpose(1,2,0))
def test(**kwargs):
    opt._parse(kwargs)
    print('load data')
    testset=TestDataset(opt)
    test_dataloader=data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")
    trainer.load('/home/huachunrui/simple_voc/checkpoints/fasterrcnn_09071710_0.26726687484801176')
   
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    imnames, gt_bboxes, gt_labels, gt_difficults = list(), list(), list(),list()
    for ii, (imgs, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(test_dataloader)):
        #print(imname,imgs.shape) 
        #print(imgs.shape,gt_bboxes_,gt_labels_)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, visualize=True)
        ori_img = visdom_bbox(at.tonumpy(imgs[0]),
                               at.tonumpy(gt_bboxes_[0]),
                               at.tonumpy(gt_labels_[0]).reshape(-1))

        ori_file=os.path.join('/home/huachunrui/simple_voc/img/'+'{}_a.jpg'.format(ii))
        imsave(ori_file,(255*at.tonumpy(ori_img)).transpose(1,2,0))

        pred_img = visdom_bbox(at.tonumpy(imgs[0]),
                               at.tonumpy(pred_bboxes_[0]),
                               at.tonumpy(pred_labels_[0]).reshape(-1),
                               at.tonumpy(pred_scores_[0]).reshape(-1))
        #print(pred_img.shape,pred_img)
        pre_file=os.path.join('/home/huachunrui/simple_voc/img/'+'{}_b.jpg'.format(ii))
        imsave(pre_file,(255*pred_img).transpose(1,2,0)) 
        if ii==5:
            break
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == opt.test_num:#132 
              np.save('/home/huachunrui/model_h5/img/img-0.78/imnames.npy',imnames)
              np.save('/home/huachunrui/model_h5/img/img-0.78/gt_bboxes.npy',gt_bboxes)
              np.save('/home/huachunrui/model_h5/img/img-0.78/gt_labels.npy',gt_labels)
              np.save('/home/huachunrui/model_h5/img/img-0.78/pred_bboxes.npy',pred_bboxes)
              np.save('/home/huachunrui/model_h5/img/img-0.78/pred_labels.npy',pred_labels)
              np.save('/home/huachunrui/model_h5/img/img-0.78/pred_scores.npy',pred_scores)


              break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    
    print("mAP: ",result['map'])
    print('Everything is ok !')





if __name__ == '__main__':
    import fire

    fire.Fire()
