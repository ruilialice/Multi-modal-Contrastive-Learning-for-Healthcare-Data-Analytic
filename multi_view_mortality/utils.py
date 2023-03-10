import torch
import pickle
import numpy as np
import torch.nn.functional as F
import os
from os.path import dirname
from collections import defaultdict
from torch.utils.data import Dataset

def contrastive_loss(code_embd, clinic_embd, label, cos, device):
    batch_size = clinic_embd.shape[0]

    v_c_loss = 0
    c_v_loss = 0
    tao = 0.2

    for b_idx in range(batch_size):
        cur_label = label[b_idx]
        nominator_flag = np.asarray([1 if cur_label==temp else 0 for temp in label])
        nominator_flag = torch.Tensor(nominator_flag).to(device)
        denominator_flag = np.asarray([1 for temp in label])
        denominator_flag = torch.Tensor(denominator_flag).to(device)
        denominator_flag[b_idx] = 0

        v_embd = code_embd[b_idx, :].unsqueeze(0).repeat((batch_size, 1))
        c_embd = clinic_embd[b_idx, :].unsqueeze(0).repeat((batch_size, 1))

        v_embd = F.normalize(v_embd, p=2, dim=1)
        c_embd = F.normalize(c_embd, p=2, dim=1)
        code_embd = F.normalize(code_embd, p=2, dim=1)
        clinic_embd = F.normalize(clinic_embd, p=2, dim=1)

        v_denominator_bi = torch.exp(cos(v_embd, clinic_embd) / tao)
        temp = denominator_flag * v_denominator_bi
        v_denominator = torch.sum(temp)
        v_nominator = torch.exp(cos(v_embd, clinic_embd) / tao)
        v_temp = torch.log(v_nominator / v_denominator)
        new_v_temp = torch.sum(nominator_flag * v_temp)
        v_c_loss -= 1 / torch.sum(nominator_flag) * new_v_temp

        c_denominator_bi = torch.exp(cos(c_embd, code_embd) / tao)
        temp = denominator_flag * c_denominator_bi
        c_denominator = torch.sum(temp)
        c_nominator = torch.exp(cos(c_embd, code_embd) / tao)
        c_temp = torch.log(c_nominator / c_denominator)
        new_c_temp = torch.sum(nominator_flag * c_temp)
        c_v_loss -= 1 / torch.sum(nominator_flag) * new_c_temp

    contrastive_loss = v_c_loss + c_v_loss
    contrastive_loss = contrastive_loss / batch_size
    return contrastive_loss

def build_tree(corMat, n_group_codes, device):
    infoDict = defaultdict(list)
    for i in range(n_group_codes):
        tmpAns = list(np.nonzero(corMat[i])[0])
        ansNum = len(tmpAns)
        tmpLea = [i] * ansNum
        infoDict[ansNum].append((tmpLea, tmpAns, i))
    lens = sorted(list(infoDict.keys()))
    leaveList = []
    ancestorList = []
    cur = 0
    mapInfo = [0] * n_group_codes
    for k in lens:
        leaves = []
        ancestors = []
        for meta in infoDict[k]:
            leaves.append(meta[0])
            ancestors.append(meta[1])
            mapInfo[meta[2]] = cur
            cur += 1
        leaves = torch.LongTensor(leaves).to(device)
        ancestors = torch.LongTensor(ancestors).to(device)
        leaveList.append(leaves)
        ancestorList.append(ancestors)
    return leaveList, ancestorList, mapInfo


def put_device(time_gru, data_gru, mask_gru, delta_gru, pad_data_gru, device):
    batch_size = len(time_gru)
    time_len = [len(temp) for temp in time_gru]
    max_len = max(time_len)
    torch_time_gru = np.zeros((batch_size, max_len))
    torch_data_gru = np.zeros((batch_size, max_len, 12))
    torch_mask_gru = np.zeros((batch_size, max_len, 12))
    torch_delta_gru = np.zeros((batch_size, max_len, 12))
    torch_pad_data_gru = np.zeros((batch_size, max_len, 12))

    for i in range(batch_size):
        torch_time_gru[i, 0:time_len[i]] = time_gru[i]
        torch_data_gru[i, 0:time_len[i], :] = data_gru[i]
        torch_mask_gru[i, 0:time_len[i], :] = mask_gru[i]
        torch_delta_gru[i, 0:time_len[i], :] = delta_gru[i]
        torch_pad_data_gru[i, 0:time_len[i], :] = pad_data_gru[i]

    torch_time_gru = torch.FloatTensor(torch_time_gru).to(device)
    torch_data_gru = torch.FloatTensor(torch_data_gru).to(device)
    torch_mask_gru = torch.FloatTensor(torch_mask_gru).to(device)
    torch_delta_gru = torch.FloatTensor(torch_delta_gru).to(device)
    torch_pad_data_gru = torch.FloatTensor(torch_pad_data_gru).to(device)
    torch_time_len = torch.LongTensor(time_len).to(device)

    return torch_time_gru, torch_data_gru, torch_mask_gru, torch_delta_gru, torch_pad_data_gru, torch_time_len

def transform(old_demographic):
    new_demographic = []

    for demographic in old_demographic:
        cur_demographic = demographic[-5:]
        cur_demo = np.zeros(12)
        cur_demo[int(cur_demographic[0])] = 1
        cur_demo[int(cur_demographic[1]) + 5] = 1
        cur_demo[9:] = cur_demographic[2:5]
        new_demographic.append(cur_demo)

    return new_demographic

def find_current_pad(cur_aug, cur_clinic_mask):
    pad_data = np.zeros(cur_aug.shape)
    time_len, feature_num = cur_clinic_mask.shape
    last_observed = np.zeros(feature_num)
    for t in range(time_len):
        temp_mask = cur_clinic_mask[t, :]
        pad_data[t] = last_observed
        pad_data[t, temp_mask] = cur_aug[t, temp_mask]
        last_observed[temp_mask] = cur_aug[t, temp_mask]
    return pad_data

class mimicDataset(Dataset):
    def __init__(self, icd9_list, demographic_list,
             clinic_time, clinic_data, clinic_mask, clinic_delta, clinic_pad_data, label, train_flag):
        self.icd9_list = icd9_list
        self.demographic_list = demographic_list
        self.clinic_time = clinic_time
        self.clinic_data = clinic_data
        self.clinic_mask = clinic_mask
        self.clinic_delta = clinic_delta
        self.clinic_pad_data = clinic_pad_data
        self.label = label
        self.train_flag = train_flag

        self.icd9_list_aug = []
        self.demographic_list_aug = []
        self.clinic_time_aug = []
        self.clinic_data_aug = []
        self.clinic_mask_aug = []
        self.clinic_delta_aug = []
        self.clinic_pad_data_aug = []
        self.label_aug = []

        if self.train_flag:
            for cur_icd, cur_demographic, cur_clinic_time, cur_clinic_data, cur_clinic_mask, cur_clinic_delta, cur_clinic_pad_data, cur_label \
                    in zip(self.icd9_list, self.demographic_list, self.clinic_time, self.clinic_data, self.clinic_mask, self.clinic_delta, self.clinic_pad_data, self.label):
                # augmentation
                if cur_label == 1:
                    for i in range(3):
                        cur_bias = np.random.normal(0, 0.1, cur_clinic_data.shape)
                        cur_bias[~cur_clinic_mask] = 0
                        cur_aug = cur_bias + cur_clinic_data
                        cur_clinic_pad_data_aug = find_current_pad(cur_aug, cur_clinic_mask)

                        self.icd9_list_aug.append(cur_icd)
                        self.demographic_list_aug.append(cur_demographic)
                        self.clinic_time_aug.append(cur_clinic_time)
                        self.clinic_data_aug.append(cur_aug)
                        self.clinic_mask_aug.append(cur_clinic_mask)
                        self.clinic_delta_aug.append(cur_clinic_delta)
                        self.clinic_pad_data_aug.append(cur_clinic_pad_data_aug)
                        self.label_aug.append(cur_label)

            self.icd9_list.extend(self.icd9_list_aug)
            self.demographic_list.extend(self.demographic_list_aug)
            self.clinic_time.extend(self.clinic_time_aug)
            self.clinic_data.extend(self.clinic_data_aug)
            self.clinic_mask.extend(self.clinic_mask_aug)
            self.clinic_delta.extend(self.clinic_delta_aug)
            self.clinic_pad_data.extend(self.clinic_pad_data_aug)
            self.label.extend(self.label_aug)

    def __len__(self):
        return len(self.icd9_list)

    def __getitem__(self, idx):
        return {'icd': self.icd9_list[idx], 'demo': self.demographic_list[idx], 'time': self.clinic_time[idx],
                'data': self.clinic_data[idx], 'mask': self.clinic_mask[idx], 'delta': self.clinic_delta[idx],
                'pad': self.clinic_pad_data[idx], 'label': self.label[idx]}

def collate_fn_padd(batch, params_now):
    device = params_now
    batch_size = len(batch)

    batch_code = [seq['icd'] for seq in batch]
    batch_medical_codes = np.zeros((batch_size, 1068))
    for idx, cur_code in enumerate(batch_code):
        batch_medical_codes[idx, cur_code] = 1
    t_medical_codes = torch.Tensor(batch_medical_codes).to(device)

    batch_label = np.asarray([seq['label'] for seq in batch])
    t_label = torch.Tensor(batch_label).to(device)

    t_profiles = [patient['demo'] for patient in batch]
    t_profiles = torch.Tensor(t_profiles).to(device)

    time_len = [len(temp['time']) for temp in batch]
    max_len = max(time_len)
    torch_time_gru = np.zeros((batch_size, max_len))
    torch_data_gru = np.zeros((batch_size, max_len, 12))
    torch_mask_gru = np.zeros((batch_size, max_len, 12))
    torch_delta_gru = np.zeros((batch_size, max_len, 12))
    torch_pad_data_gru = np.zeros((batch_size, max_len, 12))

    for i in range(batch_size):
        current_visit = batch[i]
        torch_time_gru[i, 0:time_len[i]] = current_visit['time']
        torch_data_gru[i, 0:time_len[i], :] = current_visit['data']
        torch_mask_gru[i, 0:time_len[i], :] = current_visit['mask']
        torch_delta_gru[i, 0:time_len[i], :] = current_visit['delta']
        torch_pad_data_gru[i, 0:time_len[i], :] = current_visit['pad']

    torch_time_gru = torch.Tensor(torch_time_gru).to(device)
    torch_data_gru = torch.Tensor(torch_data_gru).to(device)
    torch_mask_gru = torch.Tensor(torch_mask_gru).to(device)
    torch_delta_gru = torch.Tensor(torch_delta_gru).to(device)
    torch_pad_data_gru = torch.Tensor(torch_pad_data_gru).to(device)
    torch_time_len = torch.LongTensor(time_len).to(device)

    return t_medical_codes, t_profiles, \
           torch_time_gru, torch_data_gru, torch_mask_gru, torch_delta_gru, torch_pad_data_gru, torch_time_len, \
           t_label



def PrepareDataset():
    data_path = os.path.join(dirname(dirname(__file__)), 'data')
    train_file = 'no_outlier_gru_mimic_train_48.pkl'
    valid_file = 'no_outlier_gru_mimic_valid_48.pkl'
    test_file = 'no_outlier_gru_mimic_test_48.pkl'
    with open(os.path.join(data_path, train_file), 'rb') as f:
        total_train = pickle.load(f)
    with open(os.path.join(data_path, valid_file), 'rb') as f:
        total_valid = pickle.load(f)
    with open(os.path.join(data_path, test_file), 'rb') as f:
        total_test = pickle.load(f)

    train_time_gru, train_data_gru, train_mask_gru, train_delta_gru, train_pad_data_gru, train_label = total_train
    val_time_gru, val_data_gru, val_mask_gru, val_delta_gru, val_pad_data_gru, val_label = total_valid
    test_time_gru, test_data_gru, test_mask_gru, test_delta_gru, test_pad_data_gru, test_label = total_test

    # compute mean
    TEMP1 = np.concatenate(train_data_gru)
    TEMP2 = np.concatenate(val_data_gru)
    TEMP3 = np.concatenate(test_data_gru)
    tmask1 = np.concatenate(train_mask_gru)
    tmask2 = np.concatenate(val_mask_gru)
    tmask3 = np.concatenate(test_mask_gru)
    total_data = np.concatenate((TEMP1, TEMP2, TEMP3))
    total_mask = np.concatenate((tmask1, tmask2, tmask3))
    num_mask = np.sum(total_mask, axis=0)
    X_mean = np.sum(total_data * total_mask, axis=0) / num_mask

    train_other_file = 'no_outlier_other_mimic_train_48.pkl'
    valid_other_file = 'no_outlier_other_mimic_valid_48.pkl'
    test_other_file = 'no_outlier_other_mimic_test_48.pkl'
    with open(os.path.join(data_path, train_other_file), 'rb') as f:
        train_other = pickle.load(f)
    with open(os.path.join(data_path, valid_other_file), 'rb') as f:
        val_other = pickle.load(f)
    with open(os.path.join(data_path, test_other_file), 'rb') as f:
        test_other = pickle.load(f)

    _, old_train_demographic, _, train_icd9_short = train_other
    _, old_val_demographic, _, val_icd9_short = val_other
    _, old_test_demographic, _, test_icd9_short = test_other

    train_demographic = transform(old_train_demographic)
    valid_demographic = transform(old_val_demographic)
    test_demographic = transform(old_test_demographic)

    trainData = mimicDataset(train_icd9_short, train_demographic,
             train_time_gru, train_data_gru, train_mask_gru, train_delta_gru, train_pad_data_gru, train_label, train_flag=True)
    validData = mimicDataset(val_icd9_short, valid_demographic,
             val_time_gru, val_data_gru, val_mask_gru, val_delta_gru, val_pad_data_gru, val_label, train_flag=False)
    testData = mimicDataset(test_icd9_short, test_demographic,
            test_time_gru, test_data_gru, test_mask_gru, test_delta_gru, test_pad_data_gru, test_label, train_flag=False)

    return trainData, validData, testData, X_mean
