


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# SEED = 13
# np.random.seed(SEED)
# torch.random.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

from datasets_toy import load_toy_data

DATA_NAME = ['25gaussians', '8gaussians', 'checkerboard', '2spirals', '2circles', '2sines', '2moons', 'swissroll']
dataset_class = {'25gaussians': 25, '8gaussians': 8, 'checkerboard': 8, 
                  '2spirals': 2, '2circles': 2, '2sines': 2, '2moons': 2, 'swissroll': 1}
Network_hidden = {'25gaussians': [32, 3], '8gaussians': [16, 3], 'checkerboard': [32, 4], 
                  '2spirals': [8, 2], '2circles': [8, 2], '2sines': [8, 2], '2moons': [8, 2], 'swissroll': 'KDE'}


class Classifier(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=32, num_hidden_layer=3, num_classes=2):

        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(data_dim, hidden_dim),
                                    nn.ReLU())
        self.hid_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
                                     for i in range(num_hidden_layer)])
        self.h_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(hidden_dim//2, num_classes))

    def probability(self, y_pred):
        return nn.Softmax(dim=1)(y_pred).detach().cpu().numpy()
    
    def representation(self, x):
        h = self.layer1(x)
        for net in self.hid_layers:
            h = net(h)
        h = self.h_layer(h)
        return nn.Sigmoid()(h)
        # return h
    
    def forward(self, x):
        h = self.layer1(x)
        for net in self.hid_layers:
            h = net(h)
        h = self.h_layer(h)
        out = self.out_layer(h)
        return out

def get_dataset(dataset_name='25gaussians', num_class=25, data_size=5000):

    dataset, label = load_toy_data(dataset_name, data_size=data_size)
    X = dataset
    y = label
    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y = nn.functional.one_hot(y.long(), num_classes=num_class).to(torch.float)
    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True) 

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100   # number of epochs to run
    batch_size = 32  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    # epochs = tqdm.tqdm(range(n_epochs))
    # for epoch in range(n_epochs):
    with tqdm.tqdm(range(n_epochs), unit="batch", mininterval=0, disable=False) as bar:
        for epoch in bar:
            loss_hist = []
            model.train()
            # with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in batch_start:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                loss_hist.append(loss.item())
                # update weights
                optimizer.step()
                # print progress
                acc = (torch.argmax(y_pred, dim=-1) == torch.argmax(y_batch, dim=-1)).float().mean()
                
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_val)
            acc = (torch.argmax(y_pred, dim=-1) == torch.argmax(y_val, dim=-1)).float().mean()
            # acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
            bar.set_postfix(
                    loss=float(np.mean(loss_hist)),
                    acc=float(acc)
                )
    # restore model and return best accuracy
    # model.load_state_dict(best_weights)

    return model, acc


def get_classifier_for_evaluation(dataset_name='2spirals', num_class=25, data_size=5000, clsfr_count=1, device='cuda'):

    if not os.path.exists(f'./evaluation_classifiers'):
        save_classifier = f'./saved_result/configs_train/'
        p = Path(save_classifier)
        p.mkdir(parents=True, exist_ok=True)  
    classifier_list = []
    for i in range(clsfr_count):
        model = Classifier(num_classes=num_class).to(device)
        classifier = f'./saved_result/evaluation_classifiers/{dataset_name}_classifier_{i}.pth'

        if not os.path.exists(classifier):
            np.random.seed(1)
            torch.manual_seed(1)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True 

            X_train, X_test, y_train, y_test = get_dataset(dataset_name, num_class, data_size)
            model, acc = train_model(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
            print(f"Final model {i} accuracy: {acc*100:.2f}%")
            torch.save(model.state_dict(), classifier)

        model.load_state_dict(torch.load(classifier))
        classifier_list.append(model.eval())

    return classifier_list
       



if __name__=='__main__':

    dataset_name = '25gaussians'
    dataset, label = load_toy_data(dataset_name, data_size=2000)

    X = dataset
    y = label
    device = 'cuda'
    cv = False
    num_classes=dataset_class[dataset_name]

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y = nn.functional.one_hot(y.long(), num_classes=num_classes).to(torch.float)

    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = Classifier(num_classes=num_classes)
    print('number of parameters', sum([x.reshape(-1).shape[0] for x in model.parameters()]))  # 11041
    print('\n', model, '\n')
    # define 5-fold cross validation test harness
    if cv:
        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        cv_scores_wide = []

        cv_scores_deep = []
        for train, test in kfold.split(X_train, torch.argmax(y_train, dim=-1)):
            # create model, train, and get accuracy
            model = Classifier(num_classes=num_classes).to(device)
            acc = train_model(model, X_train[train].to(device), y_train[train].to(device), X_train[test].to(device), y_train[test].to(device))
            print("Accuracy (deep): %.2f" % acc)
            cv_scores_deep.append(acc)

        # evaluate the model
        deep_acc = np.mean(cv_scores_deep)
        deep_std = np.std(cv_scores_deep)
        print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

    model = Classifier(num_classes=num_classes).to(device)
    acc = train_model(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
    print(f"Final model accuracy: {acc*100:.2f}%")
    
    model.eval()
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            y_pred = model(X_test[i:i+1].to(device)).cpu()
            print(f"{X_test[i].numpy()} -> {torch.argmax(y_pred[0], dim=-1).numpy()} (expected {torch.argmax(y_test[i], dim=-1).numpy()})")

        # Plot the ROC curve
        y_pred = model(X.to(device)).cpu()
        acc = (torch.argmax(y_pred, dim=-1) == torch.argmax(y, dim=-1)).float().mean()
        print(f"Final model accuracy on all data: {acc*100:.2f}%")
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
        # plt.title("Receiver Operating Characteristics")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        yy = y_pred.squeeze()
        yy = torch.argmax(yy, dim=-1)
        for i in range(num_classes):
            plt.scatter(X[yy==i, 0], X[yy==i, 1], color=f'C{i}', s=20)
        plt.show()