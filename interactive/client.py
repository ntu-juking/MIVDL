import copy
import numpy as np
import sys
import torch
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score as auc ,confusion_matrix, matthews_corrcoef as mcc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tsne import plot_embedding


def train(args, model, dataset):
    model.train()
    print('Start Training', file=sys.stderr)
    #assert isinstance(model, FEDModel)
    best_f1 = 0
    best_model = None
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch_count in range(args.Epoch):
        batch_losses = []
        num_batches = dataset.initialize_train_batches()
        print("num_batches",num_batches)
        model.len = num_batches
        output_batches_generator = range(num_batches)
        for _ in output_batches_generator:
            model.zero_grad()
            features, targets = dataset.get_next_train_batch()
            probabilities, representation, batch_loss = model(example_batch=features, targets=targets)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.detach().cpu().item())

        lr_step.step()
        epoch_loss = np.mean(batch_losses).item()

        model.train()

        print('=' * 100, file=sys.stderr)
        print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=sys.stderr)
        print('=' * 100, file=sys.stderr)
        if epoch_count % 1 == 0:
            vacc, vpr, vrc, vf1, vauc = evaluate(model, dataset)
            if vf1 > best_f1:
                best_f1 = vf1
                best_model = copy.deepcopy(model)
            if sys.stderr is not None:
                print('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f \tauc: %6.3f' % \
                      (vacc, vpr, vrc, vf1, vauc), file=sys.stderr)
                print('-' * 100, file=sys.stderr)
    return best_model


def test(model,dataset):
    model.eval()
    with torch.no_grad():
        predictions = []
        expectations = []
        _batch_count = dataset.initialize_test_batches()
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            features, targets = dataset.get_next_test_batch()

            probs, _ = model(example_batch=features)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()
            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
        cm_matrix = confusion_matrix(expectations, predictions)

        class_names = [0, 1]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cm_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('.pdf')
        plt.show()
        tn, fp, fn, tp = cm_matrix.ravel()

        if (fp + tn) == 0:
            fpr = -1.0
        else:
            fpr = float(fp) / (fp + tn)

        if (tp + fn) == 0:
            fnr = -1.0
        else:
            fnr = float(fn) / (tp + fn)

        model.train()

        print('acc:', acc(expectations, predictions) * 100,
             'pre:',pr(expectations, predictions) * 100,
            'rec:',rc(expectations, predictions) * 100,
            'F1:',f1(expectations, predictions) * 100,
            'auc:',auc(expectations, predictions) * 100,
            'mcc:',mcc(expectations, predictions) * 100,
            "tp", tp,
            'fpr:',fpr * 100,
            'fnr:',fnr * 100)


def evaluate(model, dataset):
    model.eval()
    with torch.no_grad():
        predictions = [] #pred
        expectations = [] #y
        _batch_count = dataset.initialize_valid_batches()
        batch_generator = range(_batch_count)
        print("batch_generator", batch_generator)
        for _ in batch_generator:
            features, targets = dataset.get_next_valid_batch()
            print("target:",targets)
            probs, _, _ = model(example_batch=features,targets=targets)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()
            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
        print("predictions", predictions)
        print("expectations", expectations)
        # model.train()
        # return acc(expectations, predictions) * 100, \
        #        pr(expectations, predictions) * 100, \
        #        rc(expectations, predictions) * 100, \
        #        f1(expectations, predictions) * 100, \
        #        auc(expectations, predictions) * 100
        return acc(expectations, predictions), \
               pr(expectations, predictions), \
               rc(expectations, predictions), \
               f1(expectations, predictions), \
               auc(expectations, predictions)


def show_representation(model,dataset,k):
    model.eval()
    with torch.no_grad():
        representations = []
        expected_targets = []
        _batch_count = dataset.initialize_train_batches()
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            iterator_values = dataset.get_next_train_batch()
            features, targets = iterator_values[0], iterator_values[1]
            _, repr= model(example_batch=features)
            repr = repr.detach().cpu().numpy()
            #print(repr.shape)
            representations.extend(repr.tolist())
            expected_targets.extend(targets.numpy().tolist())
        model.train()
        print(np.array(representations).shape)
        print(np.array(expected_targets).shape)
        plot_embedding(representations, expected_targets, title=str(k))
