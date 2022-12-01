from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Additional Scripts
from utils import transforms as T
from utils.dataset import DentalDataset
from utils.IDRID_dataset import IDRIDDataset
from utils.utils import EpochCallback

from config import cfg

from train_transunet import TransUNetSeg

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import torch.nn.functional as F
import numpy as np


class TrainTestPipe:
    def __init__(self, train_path, test_path, model_path, lesion_type, device):
        self.device = device
        self.model_path = model_path
        self.lesion_type = lesion_type

        # self.train_loader = self.__load_dataset(train_path, train=True)
        # self.test_loader = self.__load_dataset(test_path)
        self.train_loader = self.__load_drdataset(train_path, train=True, lesion_type=self.lesion_type)
        self.test_loader = self.__load_drdataset(test_path, train=False, lesion_type=self.lesion_type)

        self.transunet = TransUNetSeg(self.device)



    def __load_drdataset(self, path, train=False, lesion_type='EX'):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            transform = transforms.Compose([T.RandomAugmentation(2)])

        set = IDRIDDataset(path, transform, lesion_type, train=train)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __load_dataset(self, path, train=False):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            transform = transforms.Compose([T.RandomAugmentation(2)])

        set = DentalDataset(path, transform)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0
        ap_value_list = []
        
        for step, data in enumerate(loader):
            img, mask = data['img'], data['mask']
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_func(img=img, mask=mask)
            ap_value_list.append(self.compute_ap(mask, cls_pred))

            total_loss += loss

            t.update()
        print("Ap value: ", str(sum(ap_value_list) / len(ap_value_list)))

        return total_loss
    
    def compute_ap(self, mask, cls_pred):
        
#         print(pred_mask.shape, params['mask'].shape)
#         print(pred_mask.max(), params['mask'].max())
#         print(pred_mask.min(), params['mask'].min())
        pred_mask_softmax_batch = F.softmax(cls_pred, dim=1).detach().cpu().numpy()
#         print(pred_mask_softmax_batch.max(), pred_mask_softmax_batch.min())

        masks_hard = mask.detach().cpu().numpy()
        
        masks_soft = np.array(pred_mask_softmax_batch).transpose((1, 0, 2, 3))
        masks_hard = np.array(masks_hard).transpose((1, 0, 2, 3))
        
        masks_soft = np.reshape(masks_soft, (masks_soft.shape[0], -1))
        masks_hard = np.reshape(masks_hard, (masks_hard.shape[0], -1))
        masks_hard = np.where(masks_hard > 0, 1, 0)
        ap = average_precision_score(masks_hard[0], masks_soft[0])
        
        return ap

    def train(self):
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', cfg.patience)

        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                train_loss = self.__loop(self.train_loader, self.transunet.train_step, t)

                test_loss = self.__loop(self.test_loader, self.transunet.test_step, t)

            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'test_loss': test_loss / len(self.test_loader)})

            if callback.end_training:
                break
