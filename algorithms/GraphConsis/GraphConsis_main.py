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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_score

import tensorflow as tf

from algorithms.GraphConsis.GraphConsis import GraphConsis
from utils.data_loader import load_data_yelp
from utils.utils import preprocess_feature
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, roc_auc_score, accuracy_score, recall_score
import matplotlib.pyplot as plt

import warnings

# Suppress this specific warning
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

# Init common args, expecting model-specific args.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8, help='training set percentage')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[10, 5], help='number of samples for each layer')
parser.add_argument('--identity_dim', type=int, default=0, help='dimension of context embedding')
parser.add_argument('--eps', type=float, default=0.001, help='consistency score threshold ε')
# New argument for adaptive temperature.
parser.add_argument('--adaptive_temp', type=float, default=1.0, help='initial adaptive temperature for sampling')

# Add to the argument parser in GraphConsis_main.py
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of attention heads for multi-head attention')

args = parser.parse_args()

# Set seeds.
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# F1 score, precision, recall, etc.
# You can use sklearn.metrics to calculate these metrics
def print_metrics(true_labels, logits, hard_preds, num_classes, stage):
    acc = accuracy_score(true_labels, hard_preds)
    prec = precision_score(true_labels, hard_preds, zero_division=0)
    recall = recall_score(true_labels, hard_preds, zero_division=0)
    f1 = f1_score(true_labels, hard_preds, zero_division=0)
    cm = confusion_matrix(true_labels, hard_preds)
    if num_classes == 2:
        # Convert logits to float32 for softmax and then compute probabilities.
        probs = tf.nn.softmax(tf.convert_to_tensor(logits.astype(np.float32))).numpy()[:, 1]
        auc = roc_auc_score(true_labels, probs)
    else:
        auc = roc_auc_score(true_labels, logits, multi_class='ovr')
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f'Recall:    {recall:.4f}')
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Create a visual display of the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{stage} Confusion Matrix')
    stage_filename = stage.replace(" ", "").replace("|", "_").replace(":", "_")
    plt.savefig(f'confusion_matrix_{stage_filename}.png')
    plt.close() 

def evaluate_in_batches(nodes, batch_size, features, neigh_dicts, cur_temp, labels, class_weights, model):
    all_logits = []  # Renamed from all_preds to all_logits
    all_true = []
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]
        batch = build_batch(batch_nodes, neigh_dicts, args.sample_sizes, features, cur_temp, labels, class_weights)
        logits = model(batch, features)  # Return logits (raw output)
        all_logits.append(logits.numpy())
        all_true.append(labels[batch_nodes])
    all_logits = np.concatenate(all_logits, axis=0)
    all_true = np.concatenate(all_true, axis=0).flatten()
    return all_true, all_logits


def GraphConsis_main(neigh_dicts, features, labels, masks, num_classes, args, class_weights):
    train_nodes, val_nodes, test_nodes = masks

    model = GraphConsis(features.shape[-1], args.nhid, len(args.sample_sizes), num_classes, len(neigh_dicts))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def generate_training_minibatch(nodes_for_training, all_labels, batch_size, features, cur_temp, class_weights):
        nodes_for_epoch = np.copy(nodes_for_training)
        ix = 0
        np.random.shuffle(nodes_for_epoch)
        while len(nodes_for_epoch) > ix + batch_size:
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            batch = build_batch(mini_batch_nodes, neigh_dicts, args.sample_sizes,
                                features, cur_temp, labels, class_weights)
            lbls = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, lbls)
        mini_batch_nodes = nodes_for_epoch[ix:]
        batch = build_batch(mini_batch_nodes, neigh_dicts, args.sample_sizes,
                            features, cur_temp, labels, class_weights)
        lbls = all_labels[mini_batch_nodes]
        yield (batch, lbls)

    for epoch in range(args.epochs):
        cur_temp = args.adaptive_temp * (0.95 ** epoch)
        print(f"\nEpoch {epoch:d}: training with adaptive temp = {cur_temp:.4f}...")
        minibatch_generator = generate_training_minibatch(train_nodes, labels,
                                                          args.batch_size, features,
                                                          cur_temp, class_weights)
        batches = len(train_nodes) / args.batch_size
        for inputs, inputs_labels in tqdm(minibatch_generator, total=batches):
            with tf.GradientTape() as tape:
                predicted = model(inputs, features)
                # calculate training metrics
                loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
                acc = accuracy_score(inputs_labels,
                                     predicted.numpy().argmax(axis=1))
                prec = precision_score(inputs_labels, predicted.numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(inputs_labels, predicted.numpy().argmax(axis=1), zero_division=0)
                f1 = f1_score(inputs_labels, predicted.numpy().argmax(axis=1), zero_division=0)
                # getting probabilities for AUC calculation
                probs = tf.nn.softmax(tf.convert_to_tensor(predicted.numpy().astype(np.float32))).numpy()[:, 1]
                auc = roc_auc_score(inputs_labels, probs, multi_class='ovr')

            grads = tape.gradient(loss, model.trainable_weights)

            # Check gradient flow
            # print("\nGradient flow check:")
            # for var, grad in zip(model.trainable_weights, grads):
            #     if grad is None:
            #         print(f"{var.name}: No gradient (None)")
            #     else:
            #         grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
            #         print(f"{var.name}: grad mean = {grad_mean:.6f}")

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}, Precision: {prec:.4f}, Recall:  {recall:.4f}, F1-score:  {f1:.4f}, AUC:  {auc:.4f}")

        # validation
        print("Validating...")
        val_results = model(build_batch(val_nodes, neigh_dicts,
                                        args.sample_sizes, features, cur_temp, labels, class_weights), features)
        loss = loss_fn(tf.convert_to_tensor(labels[val_nodes]), val_results)
        # Printing loss for validation set
        print(f" Epoch: {epoch:d}\nLoss: {loss.numpy():.4f}")
        # Printing metrics for validation set
        print_metrics(labels[val_nodes], val_results.numpy(), val_results.numpy().argmax(axis=1), num_classes, stage=f'Validation | Epoch: {epoch}')

    # testing
    print("\nTesting...")
    results = model(build_batch(test_nodes, neigh_dicts,
                                args.sample_sizes, features, cur_temp, labels, class_weights), features)
    loss = loss_fn(tf.convert_to_tensor(labels[test_nodes]), results)
    # Printing loss for test set
    print(f"Loss: {loss.numpy():.4f}")
    # Printing metrics for test set
    print_metrics(labels[test_nodes], results.numpy(), results.numpy().argmax(axis=1), num_classes, stage='Testing')
   


def build_batch(nodes: list, neigh_dicts: dict, sample_sizes: list,
                features: np.array, adaptive_temp: float, labels, class_weights) -> [namedtuple]:
    output = []
    for neigh_dict in neigh_dicts:
        dst_nodes = [nodes]
        dstsrc2dsts = []
        dstsrc2srcs = []
        dif_mats = []
        max_node_id = max(neigh_dict.keys())
        for sample_size in reversed(sample_sizes):
            ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes[-1],
                                                        neigh_dict,
                                                        sample_size,
                                                        max_node_id,
                                                        features,
                                                        adaptive_temp,
                                                        labels,
                                                        class_weights)
            dst_nodes.append(ds)
            dstsrc2srcs.append(d2s)
            dstsrc2dsts.append(d2d)
            dif_mats.append(dm)
        src_nodes = dst_nodes.pop()
        MiniBatch = namedtuple("MiniBatch", ["src_nodes", "dstsrc2srcs",
                                              "dstsrc2dsts", "dif_mats"])
        output.append(MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats))
    return output


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size,
                             max_node_id, features, adaptive_temp, labels, class_weights):
    # n - node
    # ns - neighbors of n
    def calc_consistency_score(n, ns):
        # Equation 3 in the paper
        diff = tf.norm(tf.tile([features[n]], [len(ns), 1]) - features[ns], axis=1)
        consis = tf.exp(-tf.pow(diff, 2) / adaptive_temp)
        consis = tf.where(consis > args.eps, consis, 0)
        return consis

    def sample(n, ns):
        if len(ns) == 0:
            return []
        consis = calc_consistency_score(n, ns)
        # Adjust scores using class weights.
        neighbor_labels = labels[ns].flatten()
        weight_factors = np.array([class_weights[int(l)] for l in neighbor_labels])
        consis_weighted = consis * weight_factors
        prob = consis_weighted / tf.reduce_sum(consis_weighted)
        return np.random.choice(ns, min(len(ns), sample_size), replace=False, p=prob.numpy())

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors with adaptive weighting and class bias.
    adj_mat_full = np.stack([vectorize(sample(n, neigh_dict[n]))
                             for n in dst_nodes])
    nonzero_cols_mask = np.any(adj_mat_full.astype(bool), axis=0)
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)
    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # Load the data.
    adj_list, features, split_ids, y = load_data_yelp(train_size=args.train_size)
    
    # Compute class weights for Yelpchi.
    unique, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    class_weights = {int(c): total/(len(unique)*count) for c, count in zip(unique, counts)}
    # For Yelpchi: class 0 gets ~0.585 and class 1 ~3.44, roughly.

    idx_train, _, idx_val, _, idx_test, _ = split_ids

    num_classes = len(set(y))
    label = np.array([y]).T

    features = preprocess_feature(features, to_tuple=False)
    features = np.array(features.todense())

    # Equation 2 in the paper.
    features = np.concatenate((features,
                               np.random.rand(features.shape[0], args.identity_dim)), axis=1)

    neigh_dicts = []
    for net in adj_list:
        neigh_dict = {i: [] for i in range(len(y))}
        nodes1 = net.nonzero()[0]
        nodes2 = net.nonzero()[1]
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)
        neigh_dicts.append({k: np.array(v, dtype=np.int64) for k, v in neigh_dict.items()})

    GraphConsis_main(neigh_dicts, features, label, [idx_train, idx_val, idx_test],
                     num_classes, args, class_weights)
