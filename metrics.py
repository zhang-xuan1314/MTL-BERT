import numpy as np
from sklearn.metrics import r2_score,roc_auc_score

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Records_R2(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_list = []
        self.label_list = []


    def update(self, y_pred, y_true):
        self.pred_list.append(y_pred)
        self.label_list.append(y_true)

    def results(self):
        pred = np.concatenate(self.pred_list,axis=0)
        label = np.concatenate(self.label_list,axis=0)

        results = []
        for i in range(pred.shape[1]):
            results.append(r2_score(label[:,i]*(label[:, i]!=-1000).astype('float32'),pred[:,i],sample_weight=(label[:, i]!=-1000).astype('float32')))

        return results


class Records_AUC(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_list = []
        self.label_list = []

    def update(self, y_pred, y_true):
        self.pred_list.append(y_pred)
        self.label_list.append(y_true)

    def results(self):
        pred = np.concatenate(self.pred_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)

        results = []
        for i in range(pred.shape[1]):
            results.append(roc_auc_score((label[:, i]!=-1000)*label[:, i], pred[:, i], sample_weight=(label[:, i] != -1000).astype('float32')))
        return results
