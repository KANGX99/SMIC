# import form WaIQA


#from argparse import ArgumentParser
import os,sys
import numpy as np
import random
from scipy import stats
import h5py
from PIL import Image
import torch
# from torch import nn  # modify the torch.nn.module and so on.
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision.transforms.functional import to_tensor,resize
from ignite.engine import create_supervised_evaluator
from ignite.metrics.metric import Metric
import nni
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
from nni.retiarii import model_wrapper
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
import  time

def default_loader(path, channel=3, Resize=True):
    # if channel == 1:  # 后面再改！！！
    #     return Image.open(path).convert('L')
    # else:
    assert (channel == 3)
    image=Image.open(path).convert('RGB')
    if Resize and min(image.size)>256:
        image=resize(image,256)
    image = to_tensor(image)
    return image  #

class IQADataset(Dataset):
    def __init__(self, data_info,im_dir,ref_dir,status='test', loader=default_loader):  # train改成了test
        self.status = status
        self.data_info=data_info
        self.im_dir=im_dir
        self.ref_dir=ref_dir
        # self.patch_size = args.patch_size
        # self.n_patches = args.n_patches

        Info = h5py.File(self.data_info, 'r')
        ref_ids = Info['ref_ids'][0, :]  #
        test_index = []
        for i in range(len(ref_ids)):
            test_index.append(i)

        self.index = test_index
        print("# Test Images: {}".format(len(self.index)))
        # print('Index:')
        # print(self.index)

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale  #
        self.mos=1-self.mos  # if you use LIVE, please comment this line.
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale  #
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index] - 1).astype(int)]

        self.images = ()
        self.label = []
        self.label_std = []
        # self.ims = []
        # self.refs = []
        for idx in range(len(self.index)):
            im = loader(os.path.join(self.im_dir, im_names[idx]))
            if self.ref_dir is None:
                ref = None
            else:
                ref = loader(os.path.join(self.ref_dir, ref_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

            images = im,ref
            self.images = self.images + (images,)  #

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        images = self.images[idx]
        return images, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))

# class L2pooling(nn.Module):
#     def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
#         super().__init__()
#         self.padding = (filter_size - 2 )//2
#         self.stride = stride
#         self.channels = channels
#         a = np.hanning(filter_size)[1:-1]
#         g = torch.Tensor(a[:,None]*a[None,:])
#         g = g/torch.sum(g)
#         self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
#
#     def forward(self, input):
#         input = input**2
#         out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
#         return (out+1e-12).sqrt()

@model_wrapper
class DISTS(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        self.stage1 = torch.nn.Sequential()
        # nn.ReLU(inplace=True)
        # nn.Tanh()
        # nn.LeakyReLU(0.1, inplace=True)
        # nn.Sigmoid()
        self.stage1.add_module(str(1),nn.Conv2d(3,32,3,1,1))
        self.stage1.add_module(str(2),nn.Sigmoid())
        self.stage1.add_module(str(3),nn.Conv2d(32,4096,5,1,2))
        self.stage1.add_module(str(4),nn.Sigmoid())
        self.stage1.add_module(str(5),nn.MaxPool2d(kernel_size=10, stride=5))
        seed_val = [i for i in range(-5000,5000)]
        self.seed = nn.ValueChoice(seed_val)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))  # for DISTS
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.plus_layers_num = 6
    def forward_once(self, x):
        self.activations = []
        self.activations.append(x)
        x = (x - self.mean) / self.std

        x = self.stage1(x)
        self.activations.append(x)


        return self.activations

    def distance(self, feature_ref, feature_dis, chns, plus_stage_num):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(5)
        alpha = torch.randn((1, sum(chns) + 3, 1, 1), device=device)
        beta = torch.randn((1, sum(chns) + 3, 1, 1), device=device)
        alpha.data.normal_(0.1, 0.01)
        beta.data.normal_(0.1, 0.01)
        w_sum = alpha.sum() + beta.sum()
        chns_1 = chns[:]
        chns_1.insert(0, 3)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        alpha = torch.split(alpha / w_sum, chns_1, dim=1)
        beta = torch.split(beta / w_sum, chns_1, dim=1)
        # for k in range(4 + plus_stage_num+1):
        for k in range(2):
            x_mean = feature_ref[k].mean([2, 3], keepdim=True)
            y_mean = feature_dis[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feature_ref[k] - x_mean) ** 2).mean([2, 3], keepdim=True)  # compute the variance
            y_var = ((feature_dis[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feature_dis[k] * feature_ref[k]).mean([2, 3],
                                                            keepdim=True) - x_mean * y_mean  # compute the covariance,which is equal to E(XY)-E(X)*E(Y).
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        score = 1 - (dist1 + dist2).squeeze()
        return score

    def forward(self, data):
        x, y = data
        chns_final = [4096]
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        score = self.distance(feats0, feats1, chns_final,self.plus_layers_num//3)
        # print(score)
        return score

class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE.

    `update` must receive output of the form (y_pred, y).y_pred从forward来，y从label来，这两个都未改变其形式。
    """

    def reset(self):
        self._y_pred = []
        self._y = []
        self._y_std = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        # n = int(y_pred.size(0) / y[0].size(0))  # n=1 if images; n>1 if patches
        # y_pred_im = y_pred.reshape((y[0].size(0), n)).mean(dim=1, keepdim=True)
        self._y_pred.append(y_pred.item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        # krocc = stats.stats.kendalltau(sq, q)[0]
        # plcc = stats.pearsonr(sq, q)[0]
        # rmse = np.sqrt(((sq - q) ** 2).mean())
        # mae = np.abs((sq - q)).mean()
        # outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc
               # ,krocc, plcc, rmse, mae, outlier_ratio

def get_data_loaders(data_info,im_dir,ref_dir):
    test_dataset = IQADataset(data_info,im_dir,ref_dir, 'test')  # 整个数据集
    test_loader = torch.utils.data.DataLoader(test_dataset)

    scale = test_dataset.scale

    # return train_loader, val_loader, test_loader, scale 改成下面这一句
    return test_loader, scale

def weight_init(m):
    # print(m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def evaluate_model(model_cls):
    tic = time.time()
    # parser = ArgumentParser(description='PyTorch NAS for IQA')
    # parser.add_argument("--seed", type=int, default=19920517)
    database='TID2013'
    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    if database == 'LIVE':
        data_info = './data/LIVEfullinfo.mat'
        im_dir = '/data/xiaokang/databaserelease2'
        ref_dir = '/data/xiaokang/databaserelease2/refimgs'

    if database == 'TID2013':
        data_info = './data/TID2013fullinfo.mat'
        im_dir = '/data/dataset/tid2013/distorted_images/'
        ref_dir = '/data/dataset/tid2013/reference_images/'

    # torch.manual_seed(args.seed)  #
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    model = model_cls()
    toc = time.time()
    time_used = toc-tic
    print("Creating a model from the search space took %f s"%time_used)

    # if 'NNI_OUTPUT_DIR' in os.environ:
    #     torch.onnx.export(model, (torch.randn(1, 1, 28, 28), ),
    #                       Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    tic = time.time()
    test_loader, scale = get_data_loaders(data_info,im_dir,ref_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    print("before:")
    for name, parameters in model.named_parameters():
        print(name, ':', parameters)
    print(model.seed)
    torch.manual_seed(model.seed)
    model.apply(weight_init)
    toc = time.time()
    time_used = toc-tic
    print("Loading data spend %f s"%time_used)
    print("after")
    for name, parameters in model.named_parameters():
        print(name, ':', parameters)

    # for name in model.state_dict():
        # print("{:30s}:{},device:{}".format(name, model.state_dict()[name].shape, model.state_dict()[name].device))
    tic = time.time()
    evaluator.run(test_loader)  # 把val_loader改成整个数据集
    metrics = evaluator.state.metrics
    # SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
    SROCC= metrics['IQA_performance']
    print("Test Results - SROCC: {:.4f}".format(SROCC))
    nni.report_final_result(SROCC)
    toc = time.time()
    time_used = toc-tic
    print("Test model on TID spend %f s"%time_used)

if __name__ == '__main__':
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    model_evaluator = FunctionalEvaluator(evaluate_model)

    model_space = DISTS()
    search_strategy = strategy.Random()
    exp = RetiariiExperiment(model_space, model_evaluator, [], search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'nas for IQA'
    exp_config.trial_concurrency = 1  # it can be changed according to the gpu numbers
    exp_config.max_trial_number = 2
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = False
    exp.run(exp_config, 8322)
    print('Final model:')
    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)