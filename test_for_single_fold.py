import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import torch
import torch.nn as nn

softmax = nn.Softmax(dim=1)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from my_utils.dataset import WSIDataset
from my_utils.loss import Focal_Loss, Equal_Loss, Grad_Libra_Loss, CB_loss
from my_utils.model import ABMIL, TransMIL, FCLayer, BClassifier, MILNet, PatchGCN, CLAM_SB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def set_random_seed(seed):
    random.seed(seed)  # python random module
    np.random.seed(seed)  # numpy module
    torch.manual_seed(seed)  # 为cpu设置
    torch.cuda.manual_seed(seed)  # 为当前gpu设置
    torch.cuda.manual_seed_all(seed)  # 为所有gpu设置
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.backends.cudnn.benchmark = False  # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


## 对dataloader设置随机种子：如果dataloader采用了多线程(num_workers > 1), 那么由于读取数据的顺序不同，最终运行结果也会有差异。
## 在PyTorch的DataLoader函数中为不同的work设置初始化函数，确保您的dataloader在每次调用时都以相同的顺序加载样本（随机种子固定时）
# def worker_init_fn(worker_id):
#     np.random.seed(1024 + worker_id)

def creat_dirs_for_result(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%H%M%S")
    train_name = str(timestamp) + '_' + str(args.model_name)
    log_dir = os.path.join(args.output_dir, train_name, 'logs')
    # model_dir = os.path.join(args.output_dir, train_name, 'models')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir, exist_ok=True)
    return log_dir


def save_args_json(data, log_dir):
    file = os.path.join(log_dir, 'args.json')
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)


def get_logger(log_dir, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(message)s')

    file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'train.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle = logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs = [], [], [], []

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0)

            bag = bag.to(device)
            label = label.to(device)
            output = model(bag)
            _, pred = torch.max(output, dim=1)  # 返回每一行中最大值的元素及其位置索引
            loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += loss.item()

        loss = running_loss / len(dataloader.dataset)
    return loss, slide_ids, labels, preds, probs


def CLAM_evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs = [], [], [], []

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0)

            bag = bag.to(device)
            label = label.to(device)
            output, inst_loss = model(bag, label=label, instance_eval=True)
            _, pred = torch.max(output, dim=1)  # 返回每一行中最大值的元素及其位置索引
            bag_loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += (0.7 * bag_loss.item() + 0.3 * inst_loss.item())

        loss = running_loss / len(dataloader.dataset)
    return loss, slide_ids, labels, preds, probs


def test(args, test_set):
    # 2定义dataload
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)

    # 3定义model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(args.gpus))
    print("device:{}".format(device))
    if args.model_name == 'ABMIL':
        model = ABMIL(n_classes=args.num_class, feat_size=args.feat_size).to(device)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.num_class, feat_size=args.feat_size).to(device)
    elif args.model_name == 'DSMIL':
        i_classifier = FCLayer(in_size=args.feat_size, out_size=args.num_class)
        b_classifier = BClassifier(input_size=args.feat_size, output_class=args.num_class)
        model = MILNet(i_classifier, b_classifier).to(device)
    elif args.model_name == 'CLAM':
        model = CLAM_SB(n_classes=args.num_class, feat_size=args.feat_size).to(device)
    # elif args.model_name == 'PatchGCN':
    #     model = PatchGCN(n_classes=args.num_class).to(device)
    else:
        raise NotImplementedError('no model:{}'.format(args.model_name))

    model.load_state_dict(torch.load(args.model_path))
    print("模型权重加载完成！")

    # 4定义损失函数
    if args.weighted_loss:
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(['损失权重']).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
        # Focal_Loss 测试ok
        # loss_fn = Focal_Loss(alpha=None, gamma=2, num_classes = args.num_class, size_average=True)
        # CB_loss 测试ok
        # loss_fn = CB_loss(samples_per_cls=[131,309], no_of_classes=args.num_class)
        # Equalization loss 测试ok
        # loss_fn = Equal_Loss(gamma=0.9,lambda_=0.00177,image_count_frequency=[131,309], size_average=True)
        # Equalization loss 测试ok
        # loss_fn = Grad_Libra_Loss(alpha_pos=0, alpha_neg=0, size_average=True)

    # 保存当前实验参数和结果的操作
    log_dir = creat_dirs_for_result(args)
    save_args_json(args.__dict__, log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir, name=args.logger_name)

    # 6模型训练
    kfold_test_acc = 0.0
    kfold_test_probs = []
    kfold_test_preds = []

    # test
    if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
        test_loss, test_ids, test_labels, test_preds, test_probs = evaluate(model, test_loader, loss_fn, device)
    elif args.model_name == 'CLAM':
        test_loss, test_ids, test_labels, test_preds, test_probs = CLAM_evaluate(model, test_loader, loss_fn, device)

    test_acc = accuracy_score(test_labels, test_preds)

    # tensorboard
    kfold_test_acc = test_acc
    kfold_test_preds = test_preds
    kfold_test_probs = test_probs
    logger.info('test_loss: %.4f  test_acc: %.4f' % (test_loss, test_acc))

    return kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--gpus', type=int, default=1, help="GPU indices ""comma separated, e.g. '0,1' ")
    # 'FEATURES_256_by_simclr_resnet50_1024'
    parser.add_argument('--fea_dir', type=str,
                        default='/data2/yhhu/LLB/Data/前列腺癌数据/test/features/256/uni_v1/h5_files/',
                        help='.h5 featrure files extracted by CLAM')
    parser.add_argument('--label_dir', type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/test/labels/',
                        help='splitted label file')
    parser.add_argument('--num_class', type=int, default=2, help='')
    parser.add_argument('--fold_num', type=int, default=1)
    parser.add_argument('--output_dir', default='/data2/yhhu/LLB/Data/前列腺癌数据/test/result/',
                        help='Path to experiment output, config, checkpoints, etc.')
    parser.add_argument('--logger_name', default=None, help='name of logging.getLogger(name) for record')
    # 训练相关 ['ABMIL', 'TransMIL', 'CLAM', 'DSMIL']
    parser.add_argument('--model_name', type=str, default='ABMIL', help='What model architecture to use.')
    parser.add_argument('--model_path', type=str,
                        default='/data2/yhhu/LLB/Data/前列腺癌数据/CKPan/patch/256/train_results/ABMIL/model_best.pth')
    parser.add_argument('--wd', type=float, default=0, help=' weight decay of optimizer')
    parser.add_argument('--weighted_loss', type=bool, default=False, help=' weight of loss')
    parser.add_argument('--batch_size', type=int, default=8, help='Dataloaders batch size.')
    parser.add_argument('--feat_size', default=1024, type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--model_save_num', type=int, default=2, help='Number of models saved during train.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()  # 声明中给定了各个参数的默认值
    set_random_seed(args.seed)  # 固定随机数

    # 根据特征获取feat_size，并指定output_dir
    feat_name = args.fea_dir.split('/')[-2]
    # args.feat_size = int(feat_name.split('_')[-1])
    print(args.model_name, 'feat_name:', feat_name)
    # args.output_dir = os.path.join(args.output_dir, 'kfold_test_CB_loss', str(args.fold_num) + 'fold', args.model_name + '_{}'.format(os.path.basename(args.model_path)), feat_name)

    # 1定义dataset
    df = pd.read_csv(os.path.join(args.label_dir, 'label.csv'))

    print('len(dataset):', len(df))

    # 计算结果保存
    test_acc_list, slide_ids, labels, preds, probs = [], [], [], [], []
    test_set = WSIDataset(args.fea_dir, os.path.join(args.label_dir, 'label.csv'), preload=True)
    kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs = test(args, test_set)

    test_acc_list.append(kfold_test_acc)
    slide_ids += test_ids
    labels += test_labels
    preds += kfold_test_preds
    probs += kfold_test_probs
    preds = np.array(preds)
    probs = np.array(probs)[:, 1]
    # 结果处理
    with open(os.path.join(args.output_dir, 'kfold_result.txt'), mode='w') as file:
        file.write(str(test_acc_list) + '\n')
        file.write('average {:.4f}\n'.format(sum(test_acc_list) / len(test_acc_list)))
        # file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovr")))
        # file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovo")))
        # file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovr")))
        # file.write('micro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovo")))#这种算不了
        file.write(str(classification_report(labels, preds, digits=4)))
        file.write(str(confusion_matrix(labels, preds)) + '\n')

    csv = {'slide_id': slide_ids, 'label': labels, 'probs': probs}
    csv = pd.DataFrame(csv)
    csv.to_csv(os.path.join(args.output_dir, 'prob_output.csv'), index=False)

    np.save(os.path.join(args.output_dir, 'probs.npy'), np.array(probs))
