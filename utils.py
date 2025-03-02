import math
import torch
import numpy as np
import random
import visdom
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix

class Visualizer(object):
    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]), X = np.array([x]),
                      win = str(name),
                      opts = dict(title=name),
                      update = None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img = img, win = name, opts = dict(title = name))
    def lines(self, name, line, X = None):
        if X is None:
            self.vis.line(Y = line, win = name)
        else:
            self.vis.line(X = X, Y = line, win = name)
    def scatter(self, name, data):
        self.vis.scatter(X = data, win = name)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype = np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)
    
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("auc: {:.4f}\n".format(test_info["auc"][-1]))
    fo.write("ap: {:.4f}\n".format(test_info["ap"][-1]))
    fo.write("ac: {:.4f}\n".format(test_info["ac"][-1]))
    if "far_all" in test_info:
        fo.write("far_all: {:.4f}\n".format(test_info["far_all"][-1]))


def compute_far(gt, pred, printname=None):
    preTrue = [1 if x > 0.5 else 0 for x in pred]  # 将预测标签转换为二分类的 0 或 1
    tn, fp, fn, tp = confusion_matrix(gt, preTrue).ravel()  # 计算混淆矩阵中的 TP，TN，FP，FN 值
    far = fp / (fp + tn)  # 计算 FAR
    if printname:
        print(f'far_{printname} : ' + str(far))
    return far
