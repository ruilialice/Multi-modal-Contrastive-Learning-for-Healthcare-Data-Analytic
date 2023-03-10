import torch
from GRUD import *
import _pickle as pickle
import argparse
from torch.utils.data import DataLoader
from utils import PrepareDataset, put_device, collate_fn_padd, contrastive_loss, build_tree
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score
alpha = 0
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def Train_Model(model, train_loader, valid_loader, test_loader, args, device, num_epochs=300):
    print('Model Structure: ', model)
    print('Start Training ... ')
    model_file = 'gru-d.model'+str(args.seed)

    loss_BCE = nn.BCELoss()

    learning_rate = args.lr
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    min_valid_loss = 1000

    for epoch in range(num_epochs):
        model.train()
        loss_train = 0
        loss_train_bce = 0
        loss_train_con = 0
        for idx, (t_medical_codes, t_profiles,\
           time_gru, data_gru, mask_gru, delta_gru, pad_data_gru, time_len, label) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, clinic_embd, code_embd = model(data_gru, mask_gru, delta_gru, pad_data_gru, time_len, t_medical_codes, t_profiles)
            loss_bce = loss_BCE(outputs, label)
            loss_con = contrastive_loss(code_embd, clinic_embd, label, cos, device)
            loss = loss_bce + alpha * loss_con
            loss_train_bce += loss_bce.detach().cpu().numpy()
            loss_train_con += alpha * loss_con.detach().cpu().numpy()
            loss_train += loss.detach().cpu().numpy()

            loss.backward()
            optimizer.step()
        print("current epoch: {}, total train loss: {}".format(epoch, loss_train))
        print("train loss_bce: {}, train loss con: {}".format(loss_train_bce, loss_train_con))


        # validation
        y_pred = []
        y_true = []
        loss_val_bce = 0
        loss_val_con = 0
        loss_val = 0
        with torch.no_grad():
            model.eval()
            for idx, (t_medical_codes, t_profiles, \
                      time_gru, data_gru, mask_gru, delta_gru, pad_data_gru, time_len, label) in enumerate(
                valid_loader):

                outputs_val, clinic_embd, code_embd = model(data_gru, mask_gru, delta_gru, pad_data_gru, time_len, t_medical_codes, t_profiles)
                loss_bce = loss_BCE(outputs_val, label)
                loss_con = contrastive_loss(code_embd, clinic_embd, label, cos, device)
                loss = loss_bce + alpha * loss_con
                loss_val += loss.detach().cpu().numpy()
                loss_val_bce += loss_bce.detach().cpu().numpy()
                loss_val_con += alpha * loss_con.detach().cpu().numpy()
                label = label.cpu()
                outputs_val = outputs_val.cpu()
                y_true.extend(label.data)
                y_pred.extend(outputs_val.data)

            valid_auc = roc_auc_score(y_true, y_pred)
            print("mimic3 epoch: {}, total valid loss: {}".format(epoch, loss_val))
            print("valid loss_bce: {}, valid loss con: {}".format(loss_val_bce, loss_val_con))
            print("valid_auc: {}".format(valid_auc))

            if loss_val < min_valid_loss:
                min_valid_loss = loss_val
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, model_file)
                print("saved")

            # testing the model
        with torch.no_grad():
            print("test")
            loss_test_bce = 0
            loss_test_con = 0
            loss_test = 0
            pred = []
            y_true = []
            for idx, (t_medical_codes, t_profiles, \
                      time_gru, data_gru, mask_gru, delta_gru, pad_data_gru, time_len, label) in enumerate(
                test_loader):
                outputs_test, clinic_embd, code_embd = model(data_gru, mask_gru, delta_gru, pad_data_gru, time_len, t_medical_codes, t_profiles)
                loss_bce = loss_BCE(outputs_test, label)
                loss_con = contrastive_loss(code_embd, clinic_embd, label, cos, device)
                loss = loss_bce + alpha * loss_con
                loss_test += loss.detach().cpu().numpy()
                loss_test_bce += loss_bce.detach().cpu().numpy()
                loss_test_con += alpha * loss_con.detach().cpu().numpy()
                outputs_test = outputs_test.cpu()
                label = label.cpu()
                pred.extend(outputs_test.data)
                y_true.extend(label.data)

            auc = roc_auc_score(y_true, pred)
            auprc = average_precision_score(y_true, pred)
            test_pred_binary = [0 if i <= 0.5 else 1 for i in pred]
            recall = recall_score(y_true, test_pred_binary)
            f1 = f1_score(y_true, test_pred_binary)
            print('epoch: {}, total test loss: {}'.format(epoch, loss_test))
            print("test loss_bce: {}, test loss con: {}".format(loss_test_bce, loss_test_con))
            print('test AUC: {}, test auprc: {}, test recall: {}, test f1: {}'.format(auc, auprc, recall, f1))

    print("best valid loss {}".format(min_valid_loss))
    checkpoint = torch.load(model_file)
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    # testing the model
    with torch.no_grad():
        print("load best model and test")
        loss_test = 0
        loss_test_bce = 0
        loss_test_con = 0
        pred = []
        y_true = []
        for idx, (t_medical_codes, t_profiles,\
                  time_gru, data_gru, mask_gru, delta_gru, pad_data_gru, time_len, label) in enumerate(
            test_loader):

            outputs_test, clinic_embd, code_embd = model(data_gru, mask_gru, delta_gru, pad_data_gru, time_len, t_medical_codes, t_profiles)
            loss_bce = loss_BCE(outputs_test, label)
            loss_con = contrastive_loss(code_embd, clinic_embd, label, cos, device)
            loss = loss_bce + alpha * loss_con
            loss_test += loss.detach().cpu().numpy()
            loss_test_bce += loss_bce.detach().cpu().numpy()
            loss_test_con += loss_con.detach().cpu().numpy()

            outputs_test = outputs_test.cpu()
            label = label.cpu()
            pred.extend(outputs_test.data)
            y_true.extend(label.data)

        auc = roc_auc_score(y_true, pred)
        auprc = average_precision_score(y_true, pred)
    print('test AUC: {}, AUPRC: {}'.format(auc, auprc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    # Use smaller test batch size to accommodate more importance samples
    parser.add_argument('--test-batch-size', type=int, default=256,
                        help='batch size for validation and test set')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='encoder/decoder learning rate')
    parser.add_argument('--seed', type=int, default=2021,
                        help='discriminator learning rate')
    parser.add_argument('--nz', type=int, default=32,
                        help='dimension of hidden state')


    args, _ = parser.parse_known_args()

    trainDataset, validDataset, testDataset, X_mean = PrepareDataset()
    train_loader = DataLoader(dataset=trainDataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=lambda b, params=device: collate_fn_padd(b, params),  # use custom collate function here
                          pin_memory=False)
    valid_loader = DataLoader(dataset=validDataset,
                          batch_size=args.test_batch_size,
                          shuffle=False,
                          collate_fn=lambda b, params=device: collate_fn_padd(b, params),  # use custom collate function here
                          pin_memory=False)
    test_loader = DataLoader(dataset=testDataset,
                          batch_size=args.test_batch_size,
                          shuffle=False,
                          collate_fn=lambda b, params=device: collate_fn_padd(b, params),  # use custom collate function here
                          pin_memory=False)

    X_mean = torch.FloatTensor(X_mean)
    input_dim = 12

    hierarchy_file = '../data/mimic.forgram'
    types, newfather, corMat, ini_embds = pickle.load(
        open(hierarchy_file, 'rb'))
    n_total_medical_nodes = corMat.shape[0]
    n_group_codes = 1068
    leavesList, ancestorList, mapInfo = build_tree(corMat, n_group_codes,
                                                   device)
    ini_embds = torch.FloatTensor(ini_embds).to(device)
    mapInfo = torch.LongTensor(mapInfo).to(device)
    grud = GRUD(input_dim, args, X_mean, device,
                leavesList, ancestorList, mapInfo, ini_embds,
                output_last=True).to(device)

    print("start training")

    Train_Model(grud, train_loader, valid_loader, test_loader, args, device)




