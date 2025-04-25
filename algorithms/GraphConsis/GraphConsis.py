"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2

Paper: 'Alleviating the Inconsistency Problem of
        Applying Graph Neural Network to Fraud Detection'
Link: https://arxiv.org/abs/2005.00625
"""

from collections import namedtuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import ConsisMeanAggregator
from layers.layers import ImprovedConsisMeanAggregator

init_fn = tf.keras.initializers.GlorotUniform


class GraphConsis(keras.Model):
    """
    The GraphConsis model
    """

    def __init__(self, features_dim: int, internal_dim: int, num_layers: int,
                 num_classes: int, num_relations: int, num_heads: int = 4) -> None:
        """
        :param int features_dim: input dimension
        :param int internal_dim: hidden layer dimension
        :param int num_layers: number of sample layer
        :param int num_classes: number of node classes
        :param int num_relations: number of relations
        """
        super().__init__()
        self.seq_layers = []
        self.dropouts = []
        self.attention_vec = tf.Variable(tf.random.uniform(
            [2 * internal_dim, 1], dtype=tf.float32))
        self.relation_vectors = tf.Variable(tf.random.uniform(
            [num_relations, internal_dim], dtype=tf.float32))
        for i in range(1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else features_dim
            # original aggregator layer
            # aggregator_layer = ConsisMeanAggregator(
            #     input_dim,
            #     internal_dim,
            #     name=layer_name)

            # new aggregator layer
            aggregator_layer = ImprovedConsisMeanAggregator(
                input_dim,
                internal_dim,
                num_heads=num_heads,
                num_relations=num_relations, # num_relations
                name=layer_name)
            
            self.seq_layers.append(aggregator_layer)
            self.dropouts.append(tf.keras.layers.Dropout(0.5))

        self.classifier = tf.keras.layers.Dense(num_classes,
                                                activation=tf.nn.softmax,
                                                use_bias=False,
                                                kernel_initializer=init_fn,
                                                name="classifier",
                                                )

    def call(self, minibatchs: namedtuple, features: tf.Tensor) -> tf.Tensor:
        """
        Forward propagation
        :param minibatchs: minibatch list of each relation
        :param features: 2d features of nodes
        """
        xs = []
        for i, minibatch in enumerate(minibatchs):
            x = tf.gather(tf.Variable(features, dtype=float),
                          tf.squeeze(minibatch.src_nodes))
            for j, aggregator_layer in enumerate(self.seq_layers):
                x = aggregator_layer(x,
                                     minibatch.dstsrc2srcs.pop(),
                                     minibatch.dstsrc2dsts.pop(),
                                     minibatch.dif_mats.pop(),
                                     tf.nn.embedding_lookup(
                                         self.relation_vectors, i),
                                     attention_vec = self.attention_vec,
                                     relation_vectors=self.relation_vectors, # new argument
                                     )
                x = self.dropouts[j](x)
            xs.append(x)

        return self.classifier(tf.nn.l2_normalize(
            tf.reduce_sum(
                tf.stack(xs, 1), axis=1, keepdims=False), 1)
        )
