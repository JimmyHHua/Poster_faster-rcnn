# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:41:42 2018

@author: L
"""  
import sys
sys.path.append("..")  
import numpy as np
import random  
from config import opt  
from data.dataset import Dataset 
from torch.utils import data as data_
from functools import reduce

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

class centroids_kmeans():
    def __init__(self):
        self.opt=opt
        self.n_clusters=9
        
    def iou(self,x, centroids):  
        dists = []  
        for centroid in centroids:  
            c_w, c_h = centroid  
            w, h = x  
            if c_w >= w and c_h >= h:  
                dist = w * h / (c_w * c_h)  
            elif c_w >= w and c_h <= h:  
                dist = w * c_h / (w * h + (c_w - w) * c_h)  
            elif c_w <= w and c_h >= h:  
                dist = c_w * h / (w * h + c_w * (c_h - h))  
            else:  # means both w,h are bigger than c_w and c_h respectively  
                dist = (c_w * c_h) / (w * h)  
            dists.append(dist)  
        return np.array(dists)  

    def k_means(self, x, n_clusters, eps):  
        init_index = [random.randrange(x.shape[0]) for _ in range(n_clusters)]  
        centroids = x[init_index]  
      
        d = old_d = []  
        iterations = 0  
        diff = 1e10  
        c, dim = centroids.shape  
      
        while True:  
            iterations += 1  
            d = np.array([1 - self.iou(i, centroids) for i in x])  
            if len(old_d) > 0:  
                diff = np.sum(np.abs(d - old_d))  
      
            print('diff = %f' % diff)  
      
            if diff < eps or iterations > 1000:  
                print("Number of iterations took = %d" % iterations)  
                print("Centroids = ", centroids)  
                return centroids  
      
            # assign samples to centroids  
            belonging_centroids = np.argmin(d, axis=1)  
      
            # calculate the new centroids  
            centroid_sums = np.zeros((c, dim), np.float)  
            for i in range(belonging_centroids.shape[0]):  
                centroid_sums[belonging_centroids[i]] += x[i]  
      
            for j in range(c):  
                centroids[j] = centroid_sums[j] / np.sum(belonging_centroids == j)  
      
            old_d = d.copy()

    def calc_centroids(self):
        dataset = Dataset(self.opt)
        dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)

        gt_bboxes=list()
        for img, bbox_, label_, scale in dataloader:
            gt_bboxes+=list(bbox_.numpy())
            print(len(gt_bboxes))

        X = np.array(reduce(lambda x, y: x+y, [list(_) for _ in gt_bboxes]))  #维度（6947,4）
        data_hw = np.array([X[:, 2] - X[:, 0], X[:, 3] - X[:, 1]]).T  #维度（6947,2）
        centroids=self.k_means(data_hw, self.n_clusters, 0.005)
        return centroids

if __name__=='__main__':
     a=centroids_kmeans()
     centroids=a.calc_centroids()
     #np.save("/home/xulu/model_h5/model/utils/centroids.npy",centroids)
     print(centroids)
