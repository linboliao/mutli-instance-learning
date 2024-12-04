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
from my_utils.model import ABMIL, TransMIL, FCLayer, BClassifier,MILNet, PatchGCN, CLAM_SB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def set_random_seed(seed):
    random.seed(seed)                 # python random module
    np.random.seed(seed)              # numpy module
    torch.manual_seed(seed)           # 为cpu设置
    torch.cuda.manual_seed(seed)      # 为当前gpu设置
    torch.cuda.manual_seed_all(seed)  # 为所有gpu设置
    os.environ['PYTHONHASHSEED'] = str(seed) #为了禁止hash随机化，使得实验可复现
    torch.backends.cudnn.benchmark = False  #设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
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
    train_name = str(timestamp)+'_'+str(args.model_name)
    log_dir = os.path.join(args.output_dir, train_name, 'logs')
    model_dir = os.path.join(args.output_dir, train_name, 'models')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir

def save_args_json(data, log_dir):

    file = os.path.join(log_dir, 'args.json')
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

def get_logger(log_dir, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(message)s')

    file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'train.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle=logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger

def train_one_epoch(model, train_loader, loss_fn, device, optimizer, batch_size):
    running_loss = 0.0
    model.train()

    slide_ids, preds, labels, probs= [],[],[],[]

    batch_out, batch_pred, batch_label = [],[],[]

    for i,(slide_id, bag, label) in enumerate(train_loader):
        slide_id = slide_id[0]
        bag = bag.squeeze(0)
        label = label.squeeze(0)

        bag = bag.to(device)
        label = label.to(device)
        output = model(bag)
        # print(output.shape)
        # input()
        _, pred = torch.max(output, dim=1)  #返回每一行中最大值的元素及其位置索引

        slide_ids.append(slide_id)
        preds.append(pred.detach().item())
        labels.append(label.detach().item())
        probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

        batch_out.append(output)
        batch_pred.append(pred.detach().item())
        batch_label.append(label)

        #batch批量反传
        if (i+1) % batch_size == 0 or i == len(train_loader) - 1:
            batch_out = torch.cat(batch_out)
            batch_label = torch.tensor(batch_label, device=device)

            loss = loss_fn(batch_out, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_label= [lab.detach().item() for lab in batch_label]
            print('loss: {}, label: {}, pred: {} '.format(loss.item(), batch_label, batch_pred))

            running_loss += loss.item()*len(batch_label)
            batch_out, batch_pred, batch_label = [],[],[]

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, slide_ids, labels, preds, probs

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0)

            bag = bag.to(device)
            label = label.to(device)
            output = model(bag)
            _, pred = torch.max(output, dim=1)  #返回每一行中最大值的元素及其位置索引
            loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += loss.item()

        loss = running_loss / len(dataloader.dataset)
    return loss, slide_ids, labels, preds, probs

def CLAM_train_one_epoch(model, train_loader, loss_fn, device, optimizer, batch_size):
    running_loss = 0.0
    model.train()

    slide_ids, preds, labels, probs= [],[],[],[]

    batch_out, batch_pred, batch_label, batch_inst_loss = [],[],[],[]

    for i,(slide_id, bag, label) in enumerate(train_loader):
        slide_id = slide_id[0]
        bag = bag.squeeze(0)
        label = label.squeeze(0)

        bag = bag.to(device)
        label = label.to(device)
        output, inst_loss = model(bag, label=label, instance_eval=True)
        # print(output.shape)
        # input()
        _, pred = torch.max(output, dim=1)  #返回每一行中最大值的元素及其位置索引

        slide_ids.append(slide_id)
        preds.append(pred.detach().item())
        labels.append(label.detach().item())
        probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

        batch_out.append(output)
        batch_pred.append(pred.detach().item())
        batch_label.append(label)
        batch_inst_loss.append(inst_loss)

        #batch批量反传
        if (i+1) % batch_size == 0 or i == len(train_loader) - 1:
            batch_out = torch.cat(batch_out)
            batch_label = torch.tensor(batch_label, device=device)

            bag_loss = loss_fn(batch_out, batch_label)
            inst_loss = sum(batch_inst_loss)/len(batch_inst_loss)
            loss = 0.7*bag_loss + 0.3*inst_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_label= [lab.detach().item() for lab in batch_label]
            print('bag_loss: {}, inst_loss: {}, label: {}, pred: {} '.format(bag_loss.item(), inst_loss.item(), batch_label, batch_pred))

            running_loss += loss.item()*len(batch_label)
            batch_out, batch_pred, batch_label, batch_inst_loss = [],[],[],[]

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, slide_ids, labels, preds, probs

def CLAM_evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0)

            bag = bag.to(device)
            label = label.to(device)
            output, inst_loss = model(bag, label=label, instance_eval=True)
            _, pred = torch.max(output, dim=1)  #返回每一行中最大值的元素及其位置索引
            bag_loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += (0.7*bag_loss.item() + 0.3*inst_loss.item())

        loss = running_loss / len(dataloader.dataset)
    return loss, slide_ids, labels, preds, probs

def train_and_eval(args, train_set, valid_set, test_set):

    # 2定义dataload
    train_loader = DataLoader(train_set, batch_size=1, drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)

    # 3定义model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(args.gpus))
    print("device:{}".format(device))
    if args.model_name == 'ABMIL':
        model = ABMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'DSMIL':
        i_classifier = FCLayer(in_size=args.feat_size, out_size=args.num_class)
        b_classifier = BClassifier(input_size=args.feat_size, output_class=args.num_class)
        model = MILNet(i_classifier, b_classifier).to(device)
    elif args.model_name == 'CLAM':
        model = CLAM_SB(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    # elif args.model_name == 'PatchGCN':
    #     model = PatchGCN(n_classes=args.num_class).to(device)
    else:
        raise NotImplementedError('no model:{}'.format(args.model_name))



    # 4定义损失函数
    if args.weighted_loss:
        loss_fn = nn.CrossEntropyLoss(weight= torch.FloatTensor(['损失权重']).to(device))
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

    # 5定义优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= args.epochs, eta_min=args.lr_min)

    # 保存当前实验参数和结果的操作
    log_dir, model_dir = creat_dirs_for_result(args)
    save_args_json(args.__dict__, log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir, name=args.logger_name)

    # 6模型训练
    best_acc = 0.0
    kfold_test_acc = 0.0
    kfold_test_probs = []
    kfold_test_preds = []
    model_save_list = []
    for epoch in tqdm(range(args.epochs)):
        # train
        if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
            train_loss, train_ids, train_labels, train_preds, train_probs = train_one_epoch(model, train_loader, loss_fn, device, optimizer, args.batch_size)
        elif args.model_name == 'CLAM':
            train_loss, train_ids, train_labels, train_preds, train_probs = CLAM_train_one_epoch(model, train_loader, loss_fn, device, optimizer, args.batch_size)
        train_acc = accuracy_score(train_labels, train_preds)
        print('[epoch %d] train_loss: %.3f train_acc: %.3f' % (epoch + 1, train_loss, train_acc))
        scheduler.step()

        # valid and test
        if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
            val_loss, val_ids, val_labels, val_preds, val_probs = evaluate(model, valid_loader, loss_fn, device)
            test_loss, test_ids, test_labels, test_preds, test_probs = evaluate(model, test_loader, loss_fn, device)
        elif args.model_name == 'CLAM':
            val_loss, val_ids, val_labels, val_preds, val_probs = CLAM_evaluate(model, valid_loader, loss_fn, device)
            test_loss, test_ids, test_labels, test_preds, test_probs = CLAM_evaluate(model, test_loader, loss_fn, device)

        val_acc = accuracy_score(val_labels, val_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        print('[epoch %d] val_loss: %.3f val_acc: %.3f' % (epoch + 1, val_loss, val_acc))
        print('[epoch %d] test_loss: %.3f test_acc: %.3f' % (epoch + 1, test_loss, test_acc))

        # tensorboard
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('val_loss', val_loss, epoch + 1)
        writer.add_scalar('test_loss', val_loss, epoch + 1)
        writer.add_scalar('train_acc', train_acc, epoch + 1)
        writer.add_scalar('val_acc', val_acc, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)
        logger.info('[epoch %d] train_loss: %.4f train_acc: %.4f val_loss: %.4f val_acc: %.4f test_loss: %.4f  test_acc: %.4f' %
                     (epoch + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

        #保存best_acc
        if val_acc > best_acc:
            best_acc = val_acc
            kfold_test_acc = test_acc
            kfold_test_preds = test_preds
            kfold_test_probs = test_probs

            model_save_name = 'model_e{:d}_{:.4f}.pth'.format(epoch+1, best_acc)
            model_save_list.append(model_save_name)
            torch.save(model.state_dict(), os.path.join(model_dir, model_save_name))
            #删除训练过程过多保存的模型，只保留2个
            if len(model_save_list)> args.model_save_num and os.path.isfile(os.path.join(model_dir, model_save_list[-(args.model_save_num+1)])):
                os.remove(os.path.join(model_dir, model_save_list[-(args.model_save_num+1)]))

            csv = {'slide_id':train_ids+val_ids+test_ids, 'label':train_labels+val_labels+test_labels,
                   'pred':train_preds+val_preds+test_preds, 'prob':train_probs+val_probs+test_probs}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(log_dir, 'best_epoch_output.csv'),index=False)

            with open(os.path.join(log_dir, 'best_epoch.txt'), mode='w') as file:
                file.write('{:d}\n'.format(epoch+1))
                file.write('{:.4f}\n'.format(train_acc))
                file.write('{:.4f}\n'.format(val_acc))
                file.write('{:.4f}\n'.format(test_acc))

                auc_test_probs = [e[1] for e in test_probs]
                file.write('auc {:.4f}\n'.format(roc_auc_score(test_labels, auc_test_probs)))
                file.write(str(classification_report(test_labels, test_preds)))
                file.write('train\n'+str(confusion_matrix(train_labels, train_preds))+'\n')
                file.write('val\n'+str(confusion_matrix(val_labels, val_preds))+'\n')
                file.write('test\n'+str(confusion_matrix(test_labels, test_preds))+'\n')

    # save the latest_model
    model_save_name = 'model_latest.pth'
    model_save_list.append(model_save_name)
    torch.save(model.state_dict(), os.path.join(model_dir, model_save_name))

    csv = pd.DataFrame({'model_save_list':model_save_list})
    csv.to_csv(os.path.join(model_dir, 'model_save_list.txt'),index=False)

    return kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=256, help='Random seed to use.')
    parser.add_argument('--gpus', type=int, default=0, help="GPU indices ""comma separated, e.g. '0,1' ")
    # 'FEATURES_256_by_simclr_resnet50_1024'
    parser.add_argument('--fea_dir',type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/CKPan/features/256/uni_v1/h5_files/h5_files/',help='.h5 featrure files extracted by CLAM')
    parser.add_argument('--label_dir',type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/CKPan/labels/', help='splitted label file')
    parser.add_argument('--num_class',type=int, default=2, help='')
    parser.add_argument('--fold_num',type=int, default=5)
    parser.add_argument('--output_dir', default='/data2/yhhu/LLB/Data/前列腺癌数据/CKPan/patch/256/train_results', help='Path to experiment output, config, checkpoints, etc.')
    parser.add_argument('--logger_name', default=None, help='name of logging.getLogger(name) for record')
    # 训练相关 ['ABMIL', 'TransMIL', 'CLAM', 'DSMIL']
    parser.add_argument('--model_name',type=str, default='ABMIL', help='What model architecture to use.')
    parser.add_argument('--lr', type=float, default=1e-3, help='max Learning rate of CosineAnnealingLR.')
    parser.add_argument('--lr_const', type=bool, default=True, help='max Learning rate of CosineAnnealingLR.')
    parser.add_argument('--lr_min', type=float, default=0, help='Dmin Learning rate of CosineAnnealingLR.')
    parser.add_argument('--wd', type=float, default=0, help=' weight decay of optimizer')
    parser.add_argument('--weighted_loss', type=bool, default=False, help=' weight of loss')
    parser.add_argument('--batch_size', type=int, default=8, help='Dataloaders batch size.')
    parser.add_argument('--feat_size', default=1024, type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--model_save_num', type=int, default=2, help='Number of models saved during train.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_parser()#声明中给定了各个参数的默认值
    set_random_seed(args.seed)# 固定随机数

    # 根据特征获取feat_size，并指定output_dir
    feat_name = args.fea_dir.split('/')[-3]
    # args.feat_size = 1024
    print(args.model_name, 'feat_name:', feat_name)
    args.output_dir = os.path.join(args.output_dir, 'kfold_train_val_test_CB_loss_yn', str(args.fold_num)+'fold', args.model_name, feat_name)

    # 1定义dataset
    df = pd.read_csv(os.path.join(args.label_dir, 'label.csv'))

    # 去掉列'label'的值等于3的行
    # df = df.loc[df['label'] != 3]
    # df = df.loc[df['label'] != 5]
    print('len(dataset):', len(df))

    #计算结果保存
    test_acc_list, slide_ids, labels, preds, probs = [],[],[],[],[]
    # k折数据划分与计算
    skf=StratifiedKFold(n_splits=args.fold_num, random_state=args.seed, shuffle=True)
    for i, (train_index,test_index) in enumerate(skf.split(df['slide_id'], df['label'])):

        args.logger_name = 'logger'+str(i+1)

        # temp_train_csv = os.path.join(args.label_dir, 'train_val_test_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_train.csv')
        # temp_valid_csv = os.path.join(args.label_dir, 'train_val_test_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_valid.csv')
        # temp_test_csv = os.path.join(args.label_dir, 'train_val_test_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_test.csv')
        temp_train_csv = os.path.join(args.label_dir, f'label_for_train_{i}.csv')
        temp_valid_csv = os.path.join(args.label_dir, f'label_for_valid_{i}.csv')
        temp_test_csv = os.path.join(args.label_dir, f'label_for_test_{i}.csv')
        train_df = df.iloc[train_index]
        # 训练集中分出1折做验证集，用于挑选模型
        train_df, valid_df = train_test_split(train_df, test_size=1/(args.fold_num-1), stratify=train_df['label'])
        # 将训练集、验证集、测试集暂时保存成csv文件
        train_df.to_csv(temp_train_csv,index=0)
        valid_df.to_csv(temp_valid_csv,index=0)
        df.iloc[test_index].to_csv(temp_test_csv,index=0)

        train_set = WSIDataset(args.fea_dir, temp_train_csv, preload = True)
        valid_set = WSIDataset(args.fea_dir, temp_valid_csv, preload = True)
        test_set = WSIDataset(args.fea_dir, temp_test_csv, preload = True)

        if args.lr_const:
            args.lr_min = args.lr
        else:
            args.lr_min = args.lr/10

        kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs = train_and_eval(args, train_set, valid_set, test_set)

        test_acc_list.append(kfold_test_acc)
        slide_ids += test_ids
        labels    += test_labels
        preds     += kfold_test_preds
        probs     += kfold_test_probs
        # 删除临时文件
        if os.path.isfile(temp_train_csv):
            os.remove(temp_train_csv)
        if os.path.isfile(temp_valid_csv):
            os.remove(temp_valid_csv)
        if os.path.isfile(temp_test_csv):
            os.remove(temp_test_csv)
    preds = np.array(preds)
    probs = np.array(probs)[:,1]
    # 多折交叉验证计算结束，结果处理
    with open(os.path.join(args.output_dir, 'kfold_result.txt'), mode='w') as file:
        file.write(str(test_acc_list)+'\n')
        file.write('average {:.4f}\n'.format(sum(test_acc_list)/len(test_acc_list)))
        file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovr")))
        file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovo")))
        file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovr")))
        # file.write('micro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovo")))#这种算不了
        file.write(str(classification_report(labels, preds, digits=4)))
        file.write(str(confusion_matrix(labels, preds))+'\n')

    csv = {'slide_id':slide_ids, 'label':labels, 'probs':probs}
    csv = pd.DataFrame(csv)
    csv.to_csv(os.path.join(args.output_dir, 'prob_output.csv'),index=False)

    np.save(os.path.join(args.output_dir, 'probs.npy'), np.array(probs))
