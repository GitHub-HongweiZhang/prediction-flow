from tqdm import tqdm_notebook, tqdm

import numpy as np

import torch
import torch.utils.data as data

from prediction_flow.pytorch.data import Dataset
from prediction_flow.features import Features


def __to_gpu(device, batch):
    for key, tensor in batch.items():
        batch[key] = tensor.to(device)


def fit(epochs, model, loss, optimizer, train_loader,
        valid_loader=None, notebook=False,
        auxiliary_loss_rate=0.0):
    if notebook:
        epoch_bar = tqdm_notebook(
            desc='training routine', total=epochs, position=0)
        train_bar = tqdm_notebook(
            desc='train', total=len(train_loader), position=1)
        if valid_loader:
            valid_bar = tqdm_notebook(
                desc='valid', total=len(valid_loader), position=2)
    else:
        epoch_bar = tqdm(
            desc='training routine', total=epochs, position=0)
        train_bar = tqdm(
            desc='train', total=len(train_loader), position=1)
        if valid_loader:
            valid_bar = tqdm(
                desc='valid', total=len(valid_loader), position=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        print("GPU is available, transfer model to GPU.")
        model = model.to(device)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        auxiliary_running_loss = 0
        for index, batch in enumerate(train_loader):
            if use_cuda:
                __to_gpu(device, batch)
            label = batch['label']
            # step 1. zero the gradients
            optimizer.zero_grad()
            # step 2. compute the output
            pred = model(batch)
            if isinstance(pred, tuple):
                pred, auxiliary_loss = pred
                if auxiliary_loss:
                    auxiliary_running_loss += (
                        (auxiliary_loss.item() -
                         auxiliary_running_loss) / (index + 1))
            # step 3. compute the loss
            loss_t = loss(pred, label)
            if isinstance(pred, tuple):
                if auxiliary_loss:
                    loss_t += auxiliary_loss_rate * auxiliary_loss
            running_loss += (loss_t.item() - running_loss) / (index + 1)
            # step 4. use loss to produce gradients
            loss_t.backward()
            # step 5. use optimizer to take gradient step
            optimizer.step()
            # update bar
            train_bar.set_postfix(loss=running_loss, epoch=index)
            train_bar.update()
        train_bar.reset()
        train_loss = running_loss
        train_auxiliary_loss = auxiliary_running_loss

        valid_loss = 0
        if valid_loader:
            model.eval()
            running_loss = 0
            auxiliary_running_loss = 0
            with torch.no_grad():
                for index, batch in enumerate(valid_loader):
                    if use_cuda:
                        __to_gpu(device, batch)
                    label = batch['label']
                    # step 1 compute the output
                    pred = model(batch)
                    # step 2. compute the loss
                    auxiliary_loss = torch.tensor(0.0)
                    pred = model(batch)
                    if isinstance(pred, tuple):
                        pred, auxiliary_loss = pred
                        if auxiliary_loss:
                            auxiliary_running_loss += (
                                (auxiliary_loss.item() -
                                 auxiliary_running_loss) / (index + 1))
                    loss_t = loss(pred, label)
                    if isinstance(pred, tuple):
                        if auxiliary_loss:
                            loss_t += auxiliary_loss_rate * auxiliary_loss
                    running_loss += (
                        loss_t.item() - running_loss) / (index + 1)
                    # update bar
                    valid_bar.set_postfix(
                        loss=running_loss, epoch=index)
                    valid_bar.update()
                valid_loss = running_loss
                valid_auxiliary_loss = auxiliary_running_loss
            valid_bar.reset()

        epoch_bar.set_postfix(
            train_loss=train_loss, valid_loss=valid_loss, epoch=epoch)
        epoch_bar.update()
        losses.append(
            {'train_loss': train_loss,
             'valid_loss': valid_loss,
             'train_auxiliary_loss': train_auxiliary_loss,
             'valid_auxiliary_loss': valid_auxiliary_loss})

    return losses


def predict(model, test_loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model.zero_grad()
    model.eval()

    preds = list()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            if use_cuda:
                __to_gpu(device, batch)
            # step 1 compute the output
            pred = model(batch)
            if isinstance(pred, tuple):
                pred, auxiliary_loss = pred
            preds.append(pred.cpu().numpy())

    return np.vstack(preds)


def create_dataloader_fn(
        number_features, category_features, sequence_features, batch_size,
        train_df, label_col='label', test_df=None, num_workers=0):

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    features = features.fit(train_df)

    train_X_map = features.transform(train_df)
    train_y = train_df[label_col].values
    train_dataset = Dataset(features, train_X_map, train_y)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    test_loader = None
    if test_df is not None:
        test_X_map = features.transform(test_df)
        test_y = None
        if label_col in set(test_df.columns):
            test_y = test_df[label_col].values
        test_dataset = Dataset(features, test_X_map, test_y)
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)

    return features, train_loader, test_loader
