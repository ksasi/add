import argparse
import sys
import os
import numpy as np
import librosa
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
import wandb
import itertools


parser = argparse.ArgumentParser(description='Framework for evaluating pretrained models for fake audio detection')

parser.add_argument("--track", default="LA", help='The track on which the pretrained model was trained')
parser.add_argument("--dataset_name", default="custom_dataset", help='This is the name of the dataset used for evaluation')
parser.add_argument("--dataset_path", default="/home/ubuntu/add/datasets/Dataset_Speech_Assignment", type=str, help='Path to the dataset for evaluation')
parser.add_argument("--model_path", default="/home/ubuntu/add/models/LA_model.pth", type=str, help='Path to the model files related to respective tracks (LA and DA)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size (default value is 128)')


def compute_eer_auc(labels, scores):
    """
    Ref : https://yangcha.github.io/EER-ROC/
    https://github.com/albanie/pytorch-benchmarks/blob/master/lfw_eval.py   
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer_metric = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer_metric)
    auc_metric = auc(fpr, tpr)
    return eer_metric, auc_metric, thresh

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	


class realfake_dataset(Dataset):
    def __init__(self, root_path, transforms=None):
        self.root_path = root_path
        self.transforms = transforms
        self.audio_list = glob.glob(self.root_path + '/**/*.*', recursive=True)
        self.cut=64600

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_file = self.audio_list[idx]
        label = str.split(audio_file, '/')[-2]
        audio, org_sr = librosa.load(audio_file, sr=16000)
        audio_pad = pad(audio,self.cut)
        audio_input = Tensor(audio_pad)
        if self.transforms is not None:
                audio_input = self.transforms(audio_input)
        if str.lower(label) == 'real':
            return  audio_input, torch.tensor([1])
        elif str.lower(label) == 'fake':
            return  audio_input, torch.tensor([0])

def test_epoch(model, test_loader, device):
    test_label_list = []
    test_score_list = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            model = model.to(device)
            output = model(inputs)
            score = (output[:, 1]  
                       ).data.cpu().numpy().ravel()
            labels = (labels 
                       ).data.cpu().numpy().ravel()
            test_label_list.append(labels)
            test_score_list.append(score)
    return list(itertools.chain(*test_label_list)), list(itertools.chain(*test_score_list))
    


def main():
    global args
    args = parser.parse_args()
    wandb.login()
    run = wandb.init(project="Audio_Deepfake_Detection", dir='/home/ubuntu/add/logs')

    test_dataset = realfake_dataset(root_path=args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    model = Model(args,device)
    if args.track == 'LA':
        model = Model(args,device)
    elif args.track == 'DF':
        model =nn.DataParallel(model).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('nb_params:',nb_params)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    labels, scores = test_epoch(model, test_loader, device)
    eer, auc, _ = compute_eer_auc(labels, scores)
    print(f'Evaluation metrics for {args.track} track using {args.dataset_name} :', flush=True)
    print("AUC is :", auc, flush=True)
    print("EER is :", eer, flush=True)
    wandb.finish()


if __name__ == '__main__':
    main()

    




