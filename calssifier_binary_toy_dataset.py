


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


from datasets_toy import load_toy_data


DATA_NAME = ['25gaussians', '8gaussians', 'swissroll', '2spirals', '2circles', '2sines', 'checkerboard', '2moons']


# Define two models

class Classifier_Binary(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=16, num_hidden_layer=2):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(data_dim, hidden_dim),
                                    nn.ReLU())
        self.hid_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
                                     for i in range(num_hidden_layer)])
        self.h_layer = nn.Sequential(nn.Linear(hidden_dim, 2), 
                                     nn.ReLU()) 
        self.out_layer = nn.Linear(2, 1)

        self.sigmoid = nn.Sigmoid()

    def representation(self, x):

        h = self.layer1(x)
        for net in self.hid_layers:
            h = net(h)
        out = self.h_layer(h)
        # out = self.out_layer(out)
        out = self.sigmoid(out)

        return  out
    
    def forward(self, x):
        h = self.layer1(x)
        for net in self.hid_layers:
            h = net(h)
        h = self.h_layer(h)
        out = self.out_layer(h)
        return self.sigmoid(out)


def get_dataset(dataset_name='2spirals', data_size=5000):

    dataset, label = load_toy_data(dataset_name, data_size=data_size)
    X = dataset
    y = label
    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True) 

    return X_train, X_test, y_train, y_test

# Helper function to train one model
def train_model(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100   # number of epochs to run
    batch_size = 64  # size of each batch
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
                acc = (y_pred.round() == y_batch).float().mean()
                
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                # best_weights = copy.deepcopy(model.state_dict())
            bar.set_postfix(
                    loss=float(np.mean(loss_hist)),
                    acc=float(acc)
                )
    # restore model and return best accuracy
    # model.load_state_dict(best_weights)
    return model, acc

def get_binary_classifier_for_evaluation(dataset_name='2spirals', data_size=5000, clsfr_count=1, device='cuda'):
    
    save_classifier = f'./saved_result/evaluation_classifiers/'
    if not os.path.exists(save_classifier):
        p = Path(save_classifier)
        p.mkdir(parents=True, exist_ok=True)  
    classifier_list = []
    for i in range(clsfr_count):
        model = Classifier_Binary().to(device)
        classifier = f'./saved_result/evaluation_classifiers/{dataset_name}_classifier_{i}.pth'

        if not os.path.exists(classifier):
            np.random.seed(1)
            torch.manual_seed(1)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True 

            X_train, X_test, y_train, y_test = get_dataset(dataset_name, data_size)
            model, acc = train_model(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
            print(f"Final model {i} accuracy: {acc*100:.2f}%")
            torch.save(model.state_dict(), classifier)

        model.load_state_dict(torch.load(classifier))
        classifier_list.append(model.eval())

    return classifier_list
       

# layers = []
# for module_name, module in model.named_modules():
#     if '.' not in module_name and len(module_name)>0:
#         print(f"module_name : {module_name} , value : {module}")
#         layers.append(module)





if __name__=='__main__':
    dataset, label = load_toy_data('2spirals', data_size=2000)

    X = dataset
    y = label
    device = 'cuda'
    # Binary encoding of labels
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y)

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = Classifier_Binary()
    print('number of parameters', sum([x.reshape(-1).shape[0] for x in model.parameters()]))  # 11041
    print('\n', model, '\n')
    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    cv_scores_wide = []

    cv_scores_deep = []
    for train, test in kfold.split(X_train, y_train):
        # create model, train, and get accuracy
        model = Classifier_Binary().to(device)
        model, acc = train_model(model, X_train[train].to(device), y_train[train].to(device), X_train[test].to(device), y_train[test].to(device))
        print("Accuracy: %.2f" % acc)
        cv_scores_deep.append(acc)

    # evaluate the model
    deep_acc = np.mean(cv_scores_deep)
    deep_std = np.std(cv_scores_deep)
    print("Avg acc: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

    model = Classifier_Binary().to(device)
    model, acc = train_model(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
    print(f"Final model accuracy: {acc*100:.2f}%")

    model.eval()
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            y_pred = model(X_test[i:i+1].to(device)).cpu()
            print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

        # Plot the ROC curve
        y_pred = model(X.to(device)).cpu()
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
        # plt.title("Receiver Operating Characteristics")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        yy = y_pred.round().squeeze()
        plt.scatter(X[yy==0, 0], X[yy==0, 1], color='C2', s=20)
        plt.scatter(X[yy==1, 0], X[yy==1, 1], color='C3', s=20)
        plt.show()