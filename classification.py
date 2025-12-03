from __future__ import print_function
import numpy as np
from sklearn.metrics import f1_score

def structure_imbalance_eval(dataset_str, embeds, labels, eval_iter):
    # run = wandb.init()
    from sklearn.linear_model import LogisticRegression

    import torch
    num_nodes = embeds.shape[0]

    ave_macro = []
    ave_micro = []
    with open(dataset_str + "_split_train_50.txt", 'r') as f:
        train_idx_list = f.readlines()
    with open(dataset_str + "_split_test_50.txt", 'r') as f:
        test_idx_list = f.readlines()

    for _ in range(eval_iter):

        train_mask = torch.zeros(num_nodes).bool()
        train_mask[[int(i) for i in train_idx_list[_].split(',')]] = True
        test_mask = torch.zeros(num_nodes).bool()
        test_mask[[int(i) for i in test_idx_list[_].split(',')]] = True
        clf = LogisticRegression(solver='liblinear', max_iter=400).fit(embeds[train_mask], labels[train_mask])
        output = torch.LongTensor(clf.predict(embeds[test_mask]))

        y_true = labels[test_mask].numpy()  # 将torch.Tensor转换为numpy
        y_pred = output.numpy()  # 将torch.Tensor转换为numpy
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        ave_micro.append(micro_f1)
        ave_macro.append(macro_f1)
    print(np.mean(ave_micro) * 100, '+/-', np.std(ave_micro) * 100)
    print(np.mean(ave_macro) * 100, '+/-', np.std(ave_macro) * 100)