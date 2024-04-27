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
parser.add_argument("--epochs", default=50, type=int, help='epochs for training/finetuning (default value is 50)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size (default value is 128)')
parser.add_argument("--learning_rate", default=0.01, type=float, help='initial learning rate for training (default value is 0.01)')
parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default value is 0.9)')
parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay (default value is 1e-4)')
parser.add_argument("--save_path", default="", type=str, help='path to save the checkpoint file(default is None)')
parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved model')
parser.add_argument('--loss', type=str, default='weighted_CCE')


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

def train_epoch(train_loader, model, lr, optimizer, device):
    running_loss = 0
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device, non_blocking=True)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)
        model = model.to(device, non_blocking=True)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        model = model.to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss

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

    train_dataset = realfake_dataset(root_path=args.dataset_path + '/training')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    val_dataset = realfake_dataset(root_path=args.dataset_path + '/validation')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        args.track, args.loss, args.epochs, args.batch_size, args.learning_rate)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(args.save_path , model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

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
    
    if args.track == 'LA':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.out_layer.parameters():
            param.requires_grad = True
    elif args.track == 'DF':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.out_layer.parameters():
            param.requires_grad = True

    #print(model.module.out_layer)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_epochs = args.epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.learning_rate, optimizer, device)
        val_loss = evaluate_accuracy(val_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\nEpoch: {} - Training Loss: {} - Validation Loss:{} '.format(epoch,
                                                   running_loss,val_loss), flush=True)
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))

    wandb.finish()


if __name__ == '__main__':
    main()
