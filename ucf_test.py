import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
import pickle
from utils import compute_far
warnings.filterwarnings("ignore")


def test(net, config, wind, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
            _data, _label,_name = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()
            
            res = net(_data)   
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0 :
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                
                cls_pre.append(1 if a_predict.max()>0.5 else 0)          
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:         
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  
                temp_predict = torch.zeros((0)).cuda()
   
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)

        wind.plot_lines('roc_auc', auc_score)
        wind.plot_lines('accuracy', accuracy)
        wind.plot_lines('pr_auc', ap_score)
        wind.lines('scores', frame_predict)
        wind.lines('roc_curve',tpr,fpr)
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)

def get_gt_dic(picklePath):
    with open(picklePath, 'rb') as f:
        frame_label = pickle.load(f)
    return frame_label

def pad_array(arr, length):
    """
    Pad a 1-D ndarray by the last element.
    Args:
        arr (ndarray): 1-D ndarray to be padded.
        length (int): Target length after padding.
    Returns:
        ndarray: Padded 1-D ndarray.
    """
    last_element = arr[-1]  # Get the last element
    padding_length = length - len(arr)  # Calculate the length to be padded
    if padding_length > 0:
        return np.pad(arr, (0, padding_length), 'constant', constant_values=(0, last_element))
        # Pad the array using np.pad function with padding values of 0 and the last element
    else:
        return arr  # If no padding is needed, return the original array


def test_dic(net, config, wind, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        gt_dic = get_gt_dic('./frame_label/gt-ucf-dic.pickle')
        gt, pred=[],[]  # prepare for gt_dic
        gt_abn, pred_abn=[],[]

        frame_predict = None

        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)

                _gt = gt_dic[_name[0] + '_x264']
                # 由于数据集长度不能被16整除，pred与gt不对齐，多切少补（gt）
                if len(fpre_) < len(_gt):
                    _gt = _gt[:len(fpre_)]
                else:
                    _gt = pad_array(_gt, len(fpre_))

                # gt，pred
                pred = fpre_ if pred is None else np.concatenate([pred, fpre_])
                gt = _gt if gt is None else np.concatenate([gt, _gt])
                # gt_abn，pred_abn
                if "Normal" not in _name[0]:
                    pred_abn = fpre_ if pred_abn is None else np.concatenate([pred_abn, fpre_])
                    gt_abn = _gt if gt_abn is None else np.concatenate([gt_abn, _gt])

                temp_predict = torch.zeros((0)).cuda()

        fpr, tpr, _ = roc_curve(gt, pred)
        fpr_abn, tpr_abn, _ = roc_curve(gt_abn, pred_abn)
        auc_score = auc(fpr, tpr)
        auc_abn = auc(fpr_abn,tpr_abn)
        far_all= compute_far(gt, pred)
        far_abn = compute_far(gt_abn, pred_abn)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(gt, pred,)
        ap_score = auc(recall, precision)

        wind.plot_lines('roc_auc', auc_score)
        wind.plot_lines('accuracy', accuracy)
        wind.plot_lines('pr_auc', ap_score)
        # wind.lines('scores', frame_predict)
        wind.lines('scores', pred)
        wind.lines('roc_curve', tpr, fpr)
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        test_info["far_all"].append(far_all)
        test_info["auc_abn"].append(auc_abn)
        test_info["far_abn"].append(far_abn)


