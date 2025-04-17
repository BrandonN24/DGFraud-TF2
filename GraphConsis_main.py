"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import argparse
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import tensorflow as tf

from algorithms.GraphConsis.GraphConsis import GraphConsis
from utils.data_loader import load_data_yelp
from utils.utils import preprocess_feature
import os
import warnings

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Suppress this specific warning
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

# Create a function to plot confusion matrix that can be reused
def plot_confusion_matrix(y_true, y_pred, epoch=None, phase="test"):
    # Create directory for confusion matrices if it doesn't exist
    os.makedirs(f"confusion_matrices for {args.epochs} epochs", exist_ok=True)
    
    # File name prefix based on epoch and phase
    prefix = f"{phase}_epoch_{epoch}_" if epoch is not None else f"{phase}_final_"
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print the raw confusion matrix
    print(f"{phase.capitalize()} Confusion Matrix:")
    print(cm)
    
    # Create a visual display of the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    title = f'{phase.capitalize()} Confusion Matrix'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    plt.title(title)
    plt.savefig(f'confusion_matrices for {args.epochs} epochs/{prefix}confusion_matrix.png')
    plt.close()
    
    # Normalized confusion matrix (shows percentages)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Plot the normalized confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(cmap=plt.cm.Blues, values_format='.2%')
    plt.title(f'Normalized {title}')
    plt.savefig(f'confusion_matrices for {args.epochs} epochs/{prefix}confusion_matrix_normalized.png')
    plt.close()
    
    return cm

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=6,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8,
                    help='training set percentage')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[10, 5],
                    help='number of samples for each layer')
parser.add_argument('--identity_dim', type=int, default=0,
                    help='dimension of context embedding')
parser.add_argument('--eps', type=float, default=0.001,
                    help='consistency score threshold ε')
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GraphConsis_main(neigh_dicts, features, labels, masks, num_classes, args):

    train_nodes = masks[0]
    val_nodes = masks[1]
    test_nodes = masks[2]

    # training
    def generate_training_minibatch(nodes_for_training, all_labels,
                                    batch_size, features):
        nodes_for_epoch = np.copy(nodes_for_training)
        ix = 0
        np.random.shuffle(nodes_for_epoch)
        while len(nodes_for_epoch) > ix + batch_size:
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            batch = build_batch(mini_batch_nodes, neigh_dicts,
                                args.sample_sizes, features)
            labels = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, labels)
        mini_batch_nodes = nodes_for_epoch[ix:-1]
        batch = build_batch(mini_batch_nodes, neigh_dicts,
                            args.sample_sizes, features)
        labels = all_labels[mini_batch_nodes]
        yield (batch, labels)

    model = GraphConsis(features.shape[-1], args.nhid,
                        len(args.sample_sizes), num_classes, len(neigh_dicts))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch:d}: training...")
        minibatch_generator = generate_training_minibatch(train_nodes,
                                                        labels,
                                                        args.batch_size,
                                                        features)
        batchs = len(train_nodes) / args.batch_size
        for inputs, inputs_labels in tqdm(minibatch_generator, total=batchs):

            with tf.GradientTape() as tape:
                predicted = model(inputs, features)
                loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
                acc = accuracy_score(inputs_labels,
                                    predicted.numpy().argmax(axis=1))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}")
        
        # validation - do this BEFORE trying to plot validation confusion matrix
        print("Validating...")
        val_results = model(build_batch(val_nodes, neigh_dicts,
                                        args.sample_sizes, features), features)
        loss = loss_fn(tf.convert_to_tensor(labels[val_nodes]), val_results)
        val_acc = accuracy_score(labels[val_nodes],
                                val_results.numpy().argmax(axis=1))
        print(f" Epoch: {epoch:d}, "
            f"loss: {loss.numpy():.4f}, "
            f"acc: {val_acc:.4f}")
        
        # Now plot validation confusion matrix AFTER computing val_results
        print(f"Generating validation confusion matrix for epoch {epoch}...")
        val_predictions = val_results.numpy().argmax(axis=1)
        val_true_labels = labels[val_nodes].flatten()
        plot_confusion_matrix(val_true_labels, val_predictions, epoch=epoch, phase="validation")
        
        print(f"Confusion matrix for epoch {epoch} saved in 'confusion_matrices/' directory")

    # For the testing part, you can use the plot_confusion_matrix function
    # to be consistent with the epoch validation:

    # testing
    print("Testing...")
    results = model(build_batch(test_nodes, neigh_dicts,
                                args.sample_sizes, features), features)

    # F1 score, precision, recall, etc.
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(labels[test_nodes], results.numpy().argmax(axis=1), average='weighted')
    precision = precision_score(labels[test_nodes], results.numpy().argmax(axis=1), average='weighted')
    recall = recall_score(labels[test_nodes], results.numpy().argmax(axis=1), average='weighted')
    print(f"F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # ROC-AUC score
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(labels[test_nodes], results.numpy()[:, 1])
    test_acc = accuracy_score(labels[test_nodes],
                            results.numpy().argmax(axis=1))
    print(f"Roc Auc:{roc_auc:.4f}")
    print(f"Test acc: {test_acc:.4f}")

    # Calculate and plot confusion matrix using our function
    y_true = labels[test_nodes].flatten()
    y_pred = results.numpy().argmax(axis=1)
    plot_confusion_matrix(y_true, y_pred, phase="test")

def build_batch(nodes: list, neigh_dicts: dict, sample_sizes: list,
                features: np.array) -> [namedtuple]:
    """
    :param nodes: node ids
    :param neigh_dicts: BIDIRECTIONAL adjacency matrix in dict {node:[node]}
    :param sample_sizes: sample size for each layer
    :param features: 2d features of nodes
    :return a list of namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and
        feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """

    output = []
    for neigh_dict in neigh_dicts:
        dst_nodes = [nodes]
        dstsrc2dsts = []
        dstsrc2srcs = []
        dif_mats = []

        max_node_id = max(list(neigh_dict.keys()))

        for sample_size in reversed(sample_sizes):
            ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes[-1],
                                                        neigh_dict,
                                                        sample_size,
                                                        max_node_id,
                                                        features
                                                        )
            dst_nodes.append(ds)
            dstsrc2srcs.append(d2s)
            dstsrc2dsts.append(d2d)
            dif_mats.append(dm)

        src_nodes = dst_nodes.pop()

        MiniBatchFields = ["src_nodes", "dstsrc2srcs",
                           "dstsrc2dsts", "dif_mats"]
        MiniBatch = namedtuple("MiniBatch", MiniBatchFields)
        output.append(MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats))

    return output


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size,
                             max_node_id, features):
    def calc_consistency_score(n, ns):
        # Equation 3 in the paper # sampling threshold figure out a way to adapt epsilon
        consis = tf.exp(-tf.pow(tf.norm(tf.tile([features[n]], [len(ns), 1]) -
                                        features[ns], axis=1), 2))
        consis = tf.where(consis > args.eps, consis, 0)
        return consis

    def sample(n, ns):
        if len(ns) == 0:
            return []
        consis = calc_consistency_score(n, ns)

        # Equation 4 in the paper
        prob = consis / tf.reduce_sum(consis)
        return np.random.choice(ns, min(len(ns), sample_size),
                                replace=False, p=prob)

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors
    adj_mat_full = np.stack([vectorize(sample(n, neigh_dict[n])) # change number 1
                             for n in dst_nodes])
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

    # compute diffusion matrix
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)

    # compute dstsrc mappings
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    # np.union1d automatic sorts the return,
    # which is required for np.searchsorted
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # load the data
    adj_list, features, split_ids, y = \
        load_data_yelp(train_size=args.train_size)
    idx_train, _, idx_val, _, idx_test, _ = split_ids

    num_classes = len(set(y))
    label = np.array([y]).T

    features = preprocess_feature(features, to_tuple=False)
    features = np.array(features.todense())

    # Equation 2 in the paper
    features = np.concatenate((features,
                               np.random.rand(features.shape[0],
                                              args.identity_dim)), axis=1)

    neigh_dicts = []
    for net in adj_list:
        neigh_dict = {}
        for i in range(len(y)):
            neigh_dict[i] = []
        nodes1 = net.nonzero()[0]
        nodes2 = net.nonzero()[1]
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)
        neigh_dicts.append({k: np.array(v, dtype=np.int64)
                            for k, v in neigh_dict.items()})

    GraphConsis_main(neigh_dicts, features, label,
                     [idx_train, idx_val, idx_test], num_classes, args)
