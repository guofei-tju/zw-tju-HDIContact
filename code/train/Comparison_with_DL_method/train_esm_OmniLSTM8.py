#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import random
import datetime
from sklearn import metrics
from torch import nn, optim
import pandas as pd
import torch
import xlwt,xlrd
import numpy as np
import torch.nn.functional as F
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append("/home/mqz/esm-MSA/esm-master/")
from esm.model_set import Conv_layer5,Conv_layer3,Linear_layer,BiLSTM,TurnBiLSTM,Conv_layer3_linear,Conv_layer5_linear,CrossBiLSTM
device = torch.device("cpu")
CUDA_LAUNCH_BLOCKING=1
saved_models = "/home/mqz/result/esm-feature-model/"
train_dataset_add = "/home/mqz/esm-MSA/baker31_dataset_min10.cpkl"
val_dataset_add = "/home/mqz/esm-MSA/Ecoil_valdataset_min10_2.cpkl"
test_dataset_add = "/home/mqz/esm-MSA/Ecoil_testdataset_min10_2.cpkl"
baker_attention_add = "/home/mqz/result/ImpactMSAdepth/0/baker-feature/"
ecoil_attention_add = "/home/mqz/result/ImpactMSAdepth/0/feature/"
model_type = "TurnBiLSTM256_2"#"Conv_layer5"#"Conv_layer3" TurnBiLSTM
stop_stype = "acc"#Loss
hidden_size=256
hidden_size2 = 32
num_layer=2
epochs = 800  # 200
save_name_label = model_type+"_"+stop_stype+"_esm-feature"
weight_figure = True  #loss 是否带权重
attention_dim = 660
dropout=0.1
alpha=0.1
train_dataset = pickle.load(open(train_dataset_add, 'rb'))
val_dataset = pickle.load(open(val_dataset_add, 'rb'))
test_dataset = pickle.load(open(test_dataset_add, 'rb'))
train_loss = np.zeros((epochs,1))
val_loss = np.zeros((epochs,1))
val_acc = np.zeros((epochs,1))
def printt(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))
def get_aupr(rec_ls, pre_ls):
    pr_value = 0.0
    for ix in range(len(rec_ls[:-1])):
        x_right, x_left = rec_ls[ix], rec_ls[ix + 1]
        y_top, y_bottom = pre_ls[ix], pre_ls[ix + 1]
        temp_area = abs(x_right - x_left) * (y_top + y_bottom) * 0.5
        pr_value += temp_area
    return pr_value
def testset_evaluate(preds, labels,L,real_num):
    # aupr auc top50 aupr top50auc
    labels = labels.reshape([-1])
    preds = preds.reshape([-1])
    metric = []

    # aupr 0
    precision_ls, recall_ls, thresholds = metrics.precision_recall_curve(labels, preds)
    metric.append(get_aupr(recall_ls, precision_ls))

    # auc 1
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    metric.append(metrics.auc(fpr, tpr))

    # top
    sort_result = np.zeros((2, len(labels)))
    sort_result[0, :] = preds
    sort_result[1, :] = labels
    sort_results = np.transpose(sort_result).tolist()
    sort_results.sort(reverse=True)
    sort_results = np.array(sort_results)
    # top L/5 acc 2
    metric.append(sum(sort_results[:int(L / 5), 1] == 1) / int(L / 5))
    # top L/10 acc 3
    metric.append(sum(sort_results[:int(L / 10), 1] == 1) / int(L / 10))
    # top L/20 acc 4
    metric.append( sum(sort_results[:int( L/ 20), 1]==1)  /int( L/ 20))
    # top 50 acc 5
    metric.append(sum(sort_results[:50, 1]==1)/50)
    # top 20 acc 6
    metric.append(sum(sort_results[:20, 1] == 1) / 20)
    # top 10 acc 7
    metric.append(sum(sort_results[:10, 1] == 1) / 10)
    # top 5 acc 8
    metric.append(sum(sort_results[:5, 1] == 1) / 5)
    # top real acc 9
    metric.append(sum(sort_results[:real_num, 1] == 1) / real_num)
    # 10
    RPFF = np.where(sort_results[:, 1] == 1)[0][0]
    metric.append(int(RPFF) + 1)
    return metric
def normalization(data):
    _range = np.max(data) - np.min(data)
    new_data = (data - np.min(data)) / _range
    return new_data
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduce=False)
        self.gamma = gamma

    def forward(self, pred, label, weight_mask):
        BCE_loss = self.criterion(pred, label) # = -log(pt)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        loss_weight = F_loss.mul(weight_mask)
        loss = torch.mean(loss_weight)
        return loss

# 加载模型
if model_type == "Conv_layer3":
    model = Conv_layer3(attention_dim,act=F.sigmoid,act_true = False)
elif model_type == "Conv_layer5":
    model = Conv_layer5(attention_dim, act=F.sigmoid, act_true=False)
elif model_type == "Conv_layer3_linear":
    model = Conv_layer3_linear(attention_dim, hidden_size, act=F.sigmoid, act_true=False)
elif model_type == "Conv_layer5_linear":
    model = Conv_layer5_linear(attention_dim, hidden_size, act=F.sigmoid, act_true=False)
elif model_type.startswith("BiLSTM"):
    model = BiLSTM(attention_dim,hidden_size,num_layer,act=F.sigmoid,act_true = False)
elif model_type.startswith("TurnBiLSTM"):
    model = TurnBiLSTM(attention_dim,hidden_size,num_layer,act=F.sigmoid,act_true = False)
elif model_type.startswith("CrossBiLSTM"):
    model = CrossBiLSTM(attention_dim,hidden_size,hidden_size2,num_layer,act=F.sigmoid,act_true = False)
else:
    model = Linear_layer(attention_dim, act=F.sigmoid, act_true=False)
model.to(device)  # 将模型加载到指定设备上
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
criterion=FocalLoss()

es = 0
best_avg_loss = 99999.
best_acc = 0.
best_model_add = ""
for e in range(epochs):  # 循环训练
    e_loss = 0.  # 当前epochs loss
    for protein in train_dataset:
        pdb_id = protein["Complex_code"]
        L = protein["Ligand_length"]
        E = pickle.load(open(baker_attention_add+pdb_id+"_attentions.pkl", 'rb'))[:L,L:]
        #E归一化
        for E_feature_num in range(np.size(E,2)):
            E[:,:,E_feature_num] = normalization(E[:,:,E_feature_num])
        Map_Label = protein["Map_Label"][:L,L:]
        pscore = ( Map_Label.shape[0]*Map_Label.shape[1]-np.sum(np.sum(Map_Label==1)) )/np.sum(np.sum(Map_Label==1))
        if weight_figure:
            Label_weight = np.where(Map_Label>0,pscore,1)
        else:
            Label_weight = np.ones(Map_Label.shape)
        L = torch.tensor(L).to(device)
        E = torch.FloatTensor(E).to(device)
        Map_Label = torch.FloatTensor(Map_Label).to(device)
        Label_weight = torch.FloatTensor(Label_weight).to(device)
        optimizer.zero_grad()  # 梯度设为0
        protein_preds = model(E)
        pr_preds = protein_preds
        pr_labels = Map_Label
        pr_weight = Label_weight
        protein_loss=criterion(pr_preds,pr_labels,pr_weight)
        pr_loss = protein_loss.item()  # it_loss(batch_loss) iter_losses(all batch_loss list)
        protein_loss.backward()  # 反向传播计算
        optimizer.step()  # 优化器根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
        e_loss += pr_loss

    e_loss = e_loss / (len(train_dataset))
    train_loss[e, 0] = e_loss
    if (e + 1) % 10 == 0:
        printt("epochs: {} loss: {}".format(e + 1, e_loss))

    val_avg_acc = 0. #top real acc
    val_avg_loss = 0. #
    met_all = []

    for protein in val_dataset:
        pdb_id = protein["Complex_code"]
        L = protein["l_length"]
        E = pickle.load(open(ecoil_attention_add + pdb_id + "_attentions.pkl", 'rb'))[:L, L:]
        # E归一化
        for E_feature_num in range(np.size(E, 2)):
            E[:, :, E_feature_num] = normalization(E[:, :, E_feature_num])
        Map_Label = protein["Map_Label"][:L, L:]
        p_num = np.sum(np.sum(Map_Label == 1))
        pscore = (Map_Label.shape[0] * Map_Label.shape[1] - p_num) / p_num
        if weight_figure:
            Label_weight = np.where(Map_Label > 0, pscore, 1)
        else:
            Label_weight = np.ones(Map_Label.shape)
        L = torch.tensor(L).to(device)
        E = torch.FloatTensor(E).to(device)
        Map_Label = torch.FloatTensor(Map_Label).to(device)
        Label_weight = torch.FloatTensor(Label_weight).to(device)
        protein_preds = model(E)
        pr_l = Map_Label.cpu().numpy().flatten()
        pr_p = F.sigmoid(protein_preds).detach().cpu().numpy().flatten()
        protein_loss = criterion(protein_preds, Map_Label, Label_weight)  #
        pr_loss = protein_loss.item()
        met = testset_evaluate(pr_p, pr_l, protein["l_length"],p_num)
        val_avg_acc += met[9]
        val_avg_loss += pr_loss
        mett = dict()
        mett["Name"] = pdb_id
        mett["Value"] = met
        met_all.append(mett)
    val_avg_acc = val_avg_acc / (len(val_dataset))
    val_avg_loss = val_avg_loss / (len(val_dataset))
    val_loss[e, 0] = val_avg_loss
    val_acc[e, 0] = val_avg_acc
    Early_stopping_Flag = False
    if stop_stype == "Loss":
        if val_avg_loss < best_avg_loss:
            Early_stopping_Flag = False
        else:
            Early_stopping_Flag = True
    else:
        if val_avg_acc > best_acc:
            Early_stopping_Flag = False
        else:
            Early_stopping_Flag = True

    if Early_stopping_Flag==False:
        best_acc = val_avg_acc
        best_avg_loss = val_avg_loss
        es = 0
        delete_add = os.path.join(saved_models, "bakertrain_"+save_name_label+"_*.tar")
        torch.save(model.state_dict(), os.path.join(saved_models, "bakertrain_"+save_name_label+"_{}.tar".format(e)))
        best_model_add = os.path.join(saved_models, "bakertrain_"+save_name_label+"_{}.tar".format(e))
        printt("UPDATE\tEpoch {}: val avg loss{} val avg acc{}".format(e + 1, best_avg_loss, best_acc))
    else:
        es += 1
        print("Counter {} of 5".format(es))

        if es > 10:
            print("Early stopping with best_acc: ", best_acc)
            break

# train record
data_trainloss = pd.DataFrame(train_loss)
data_valloss = pd.DataFrame(val_loss)
data_valacc = pd.DataFrame(val_acc)
writer = pd.ExcelWriter(os.path.join(saved_models, "bakertrain_"+save_name_label+"_epochrecord.xlsx"))  # 写入Excel文件
data_trainloss.to_excel(writer, 'trainloss', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
data_valloss.to_excel(writer, 'valloss', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
data_valacc.to_excel(writer, 'valacc', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()
writer.close()


#val information save
book = xlwt.Workbook()
sheet = book.add_sheet('Sheet1')
sheet.write(0, 0, "Complex_code")
sheet.write(0, 1, "aupr")
sheet.write(0, 2, "auc")
sheet.write(0, 3, "top L/5 acc")  #
sheet.write(0, 4, "top L/10 acc")  #
sheet.write(0, 5, "top L/20 acc")  #
sheet.write(0, 6, "top 50 acc")
sheet.write(0, 7, "top 20 acc")
sheet.write(0, 8, "top 10 acc")
sheet.write(0, 9, "top 5 acc")
sheet.write(0, 10, "top real acc")
sheet.write(0, 11, "REFF")
for t, met_context in enumerate(met_all):
    sheet.write(t + 1, 0, met_context["Name"])
    for p in range(len(met_context["Value"])):
        sheet.write(t + 1, p + 1, met_context["Value"][p])
book.save(os.path.join(saved_models, "bakertrain_"+save_name_label+"_valEcoil.xlsx"))

#test
model.load_state_dict(torch.load(best_model_add))
torch.no_grad()
met_all = []
test_avg_acc = 0.
for protein in test_dataset:
    pdb_id = protein["Complex_code"]
    L = protein["l_length"]
    E = pickle.load(open(ecoil_attention_add + pdb_id + "_attentions.pkl", 'rb'))[:L, L:]
    # E归一化
    for E_feature_num in range(np.size(E, 2)):
        E[:, :, E_feature_num] = normalization(E[:, :, E_feature_num])
    Map_Label = protein["Map_Label"][:L, L:]
    p_num = np.sum(np.sum(Map_Label == 1))
    L = torch.tensor(L).to(device)
    E = torch.FloatTensor(E).to(device)
    Map_Label = torch.FloatTensor(Map_Label).to(device)
    protein_preds = model(E)
    pr_l = Map_Label.cpu().numpy().flatten()
    pr_p = F.sigmoid(protein_preds).detach().cpu().numpy()
    #np.savetxt(os.path.join("F:/paper2exper/Ecoil_test_result/ecoil_our_result/", protein["Complex_code"] + ".txt"), pr_p)
    met = testset_evaluate(pr_p.flatten(), pr_l, protein["l_length"], p_num)
    test_avg_acc += met[9]
    mett = dict()
    mett["Name"] = pdb_id
    mett["Value"] = met
    met_all.append(mett)
test_avg_acc = test_avg_acc / (len(test_dataset))

printt("Ending\tEpoch {}: test avg acc{}".format(e + 1, test_avg_acc))
book = xlwt.Workbook()
sheet = book.add_sheet('Sheet1')
sheet.write(0, 0, "Complex_code")
sheet.write(0, 1, "aupr")
sheet.write(0, 2, "auc")
sheet.write(0, 3, "top L/5 acc")  #
sheet.write(0, 4, "top L/10 acc")  #
sheet.write(0, 5, "top L/20 acc")  #
sheet.write(0, 6, "top 50 acc")
sheet.write(0, 7, "top 20 acc")
sheet.write(0, 8, "top 10 acc")
sheet.write(0, 9, "top 5 acc")
sheet.write(0, 10, "top real acc")
sheet.write(0, 11, "REFF")
for t, met_context in enumerate(met_all):
    sheet.write(t + 1, 0, met_context["Name"])
    for p in range(len(met_context["Value"])):
        sheet.write(t + 1, p + 1, met_context["Value"][p])
book.save(os.path.join(saved_models, "bakertrain_"+save_name_label+"_testEcoil.xlsx"))
