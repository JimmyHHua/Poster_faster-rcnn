'''
import  numpy as np
import random
c=np.load('centroids.npy')

init_index = [random.randrange(20) for _ in range(9)]
print(init_index)
'''

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


dataset = Dataset(opt)
dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
print(len(dataloader))

gt_bboxes=[]
for img, bbox_, label_, scale in dataloader:
    gt_bboxes+=list(bbox_.numpy())
   #print(len(gt_bboxes))

X = np.array(reduce(lambda x, y: x+y, [list(_) for _ in gt_bboxes])) #（6947,4）

print(X.shape)  
data_hw = np.array([X[:, 2] - X[:, 0], X[:, 3] - X[:, 1]]).T  #（6947,2）
print(data_hw.shape)
