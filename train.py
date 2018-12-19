import os

import ipdb
import matplotlib
import torch as t
from tqdm import tqdm

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


def eval(dataloader, faster_rcnn, test_num=133):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imname, imgs, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        #print(imname,imgs.shape) 
        #print(imgs.shape,gt_bboxes_,gt_labels_)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, visualize=True)
    # for ii, (imname,imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
    #     sizes = [sizes[0][0], sizes[1][0]]
    #     pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    print('the labels is :',dataset.db.label_names)
    #trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(14):
        trainer.reset_meters()
        print('hello')

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            #print('hahah')

            # if ii==5:
            #     break

            # if (ii + 1) % opt.plot_every == 0:
            #     if os.path.exists(opt.debug_file):
            #         ipdb.set_trace()

            #     # plot loss
            #     #trainer.vis.plot_many(trainer.get_meter_data())
            #     #print(trainer.get_meter_data())
            #     # plot groud truth bboxes
            #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = visdom_bbox(ori_img_,
            #                          at.tonumpy(bbox_[0]),
            #                          at.tonumpy(label_[0]))
            #     trainer.vis.img('gt_img', gt_img)

            #     # plot predicti bboxes
            #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            #     pred_img = visdom_bbox(ori_img_,
            #                            at.tonumpy(_bboxes[0]),
            #                            at.tonumpy(_labels[0]).reshape(-1),
            #                            at.tonumpy(_scores[0]))
            #     trainer.vis.img('pred_img', pred_img)

            #     # rpn confusion matrix(meter)
            #     #trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            #     # roi confusion matrix
            #     trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^') 
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        print('map:',eval_result['map'],'best_map:',best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        print('*************************')
        #trainer.vis.plot('test_map', eval_result['map'])
        # log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
        #                                           str(eval_result['map']),
        #                                           str(trainer.get_meter_data()))
        # #trainer.vis.log(log_info)
        print('***************: this is epoch: ', epoch)
       # if epoch == 1: 
           # break
def test_me(**kwargs):

    opt._parse(kwargs)
    img=read_image('/home/huachunrui/model_h5/img/tim.jpg')
    print('1: ',img.shape)
    img=t.from_numpy(img)[None]
    print('2: ',img.shape)
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")
    trainer.load(opt.load_path)
    pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, visualize=True)
    print('img numpy is: ',at.tonumpy(img[0]))
    pred_img = visdom_bbox(at.tonumpy(img[0]),
                               at.tonumpy(pred_bboxes_[0]),
                               at.tonumpy(pred_labels_[0]).reshape(-1),
                               at.tonumpy(pred_scores_[0]).reshape(-1))
    imsave('/home/huachunrui/model_h5/img/000b.jpg',(255*pred_img).transpose(1,2,0))

    print('pass')

def test_jimmy(**kwargs):
    opt._parse(kwargs)
    print('load data')
    testset=TestDataset(opt)
    test_dataloader=data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")
    trainer.load(opt.load_path)
   
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    imnames, gt_bboxes, gt_labels, gt_difficults = list(), list(), list(),list()
    for ii, (imname, imgs, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(test_dataloader)):
        #print(imname,imgs.shape) 
        #print(imgs.shape,gt_bboxes_,gt_labels_)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, visualize=True)
        ori_img = visdom_bbox(at.tonumpy(imgs[0]),
                               at.tonumpy(gt_bboxes_[0]),
                               at.tonumpy(gt_labels_[0]).reshape(-1))

        ori_file=os.path.join('/home/huachunrui/model_h5/img/img-0.78/'+'{}_origin.jpg'.format(ii))
        imsave(ori_file,(255*at.tonumpy(ori_img)).transpose(1,2,0))

        pred_img = visdom_bbox(at.tonumpy(imgs[0]),
                               at.tonumpy(pred_bboxes_[0]),
                               at.tonumpy(pred_labels_[0]).reshape(-1),
                               at.tonumpy(pred_scores_[0]).reshape(-1))
        #print(pred_img.shape,pred_img)
        pre_file=os.path.join('/home/huachunrui/model_h5/img/img-0.77/'+'{}_detected.jpg'.format(ii))
        imsave(pre_file,(255*pred_img).transpose(1,2,0)) 
        if ii==2:
            break
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        imnames += imname
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


def test(**kwargs):
    opt._parse(kwargs)
    dataset=Dataset(opt)
    print('load data')
    testset=TestDataset(opt)
    test_dataloader=data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)
    faster_rcnn=FasterRCNNVGG16()
    print('model construct completed')
    trainer=FasterRCNNTrainer(faster_rcnn).cuda()
    print("load all weights")

    #opt.load_path='/home/xulu/model_h5/checkpoints/fasterrcnn_04251612_0.784629387622'

    trainer.load(opt.load_path)
    #if opt.load_path:
    #   trainer.load(opt.load_path)
    #   print('load pretrained model from %s'% opt.load_path)
   
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    imnames, gt_bboxes, gt_labels, gt_difficults = list(), list(), list(),list()
    for ii, (imname, imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(test_dataloader)):
        print(imname,imgs[0].shape)
       # print(imgs.shape,gt_bboxes_,gt_labels_)
        sizes = [sizes[0][0], sizes[1][0]]
       # print(sizes)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
       # print(pred_bboxes_, pred_labels_, pred_scores_)
	# plot groud truth bboxes
        ori_img_ = inverse_normalize(at.tonumpy(imgs[0]))
        imsave('/home/huachunrui/result_test_h5/a.jpg',(255*ori_img_).transpose(1,2,0))
        gt_img = visdom_bbox(ori_img_,
                             at.tonumpy(gt_bboxes_[0]),
                             at.tonumpy(gt_labels_[0]))
       # print(gt_img.shape)
        imsave('/home/huachunrui/result_test_h5/b.jpg',(255*gt_img).transpose(1,2,0))
        # plot predicti bboxes
        pred_img = visdom_bbox(gt_img,
                               at.tonumpy(pred_bboxes_[0]),
                               at.tonumpy(pred_labels_[0]).reshape(-1),
                               at.tonumpy(pred_scores_[0]))
       # print(pred_img.shape,pred_img)
        imsave('/home/huachunrui/result_test_h5/c.jpg',(255*pred_img).transpose(1,2,0)) 
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        imnames += imname
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == opt.test_num:#132 
              np.save('/home/huachunrui/result_test_h5/imnames.npy',imnames)
              np.save('/home/huachunrui/result_test_h5/gt_bboxes.npy',gt_bboxes)
              np.save('/home/huachunrui/result_test_h5/gt_labels.npy',gt_labels)
              np.save('/home/huachunrui/result_test_h5/pred_bboxes.npy',pred_bboxes)
              np.save('/home/huachunrui/result_test_h5/pred_labels.npy',pred_labels)
              np.save('/home/huachunrui/result_test_h5/pred_scores.npy',pred_scores)


              break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    
    print("mAP: ",result['map'])





if __name__ == '__main__':
    import fire

    fire.Fire()
