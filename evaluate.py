"""Evaluate the model"""

import argparse
import logging
import os

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from tools import utils
import model.net as net
from tools.data_loader import DataLoader


def evaluate(model, data_iterator, num_steps, metric_labels):
    """Evaluate the model on `num_steps` batches."""
    # set model to evaluation mode
    model.eval()

    output_labels = list()
    target_labels = list()

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        batch_data, batch_labels = next(data_iterator)

        # compute model output
        batch_output = model(batch_data)  # batch_size x num_labels
        batch_output_labels = torch.max(batch_output, dim=1)[1]
        output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
        target_labels.extend(batch_labels.data.cpu().numpy().tolist())

    # Calculate precision, recall and F1 for all relation categories
    p_r_f1_s = precision_recall_fscore_support(
        target_labels, output_labels, labels=metric_labels, average='micro')
    p_r_f1 = {'precison': p_r_f1_s[0] * 100,
              'recall': p_r_f1_s[1] * 100,
              'f1': p_r_f1_s[2] * 100}
    return p_r_f1
