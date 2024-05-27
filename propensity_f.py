from itertools import chain

import numpy as np
from collections import defaultdict, Counter


class PropensityF:
    def __init__(self, labels, A=0.55, B=1.5):
        """
        Инициализация класса с множеством меток и параметрами A и B.
        """
        self.labels = labels | {0}
        self.A = A
        self.B = B

    def calculate_prop_f(self, _y_true, _y_pred):
        y_true = [labels_set | {0} for labels_set in _y_true]
        y_pred = [labels_set | {0} for labels_set in _y_pred]

        label_propensities = self.compute_propensities(y_true)

        propensity_score = lambda labels: sum(1 / label_propensities[label] for label in labels if label in label_propensities)

        prop_precisions = []
        prop_recalls = []

        for true_labels, pred_labels in zip(y_true, y_pred):
            if len(pred_labels) == 0:
                prop_precision = 0
            else:
                prop_precision = propensity_score(pred_labels & true_labels) / propensity_score(pred_labels)

            if len(true_labels) == 0:
                prop_recall = 0
            else:
                prop_recall = propensity_score(pred_labels & true_labels) / propensity_score(true_labels)

            prop_precisions.append(prop_precision)
            prop_recalls.append(prop_recall)

        prop_precision_avg = np.mean(prop_precisions)
        prop_recall_avg = np.mean(prop_recalls)

        if prop_precision_avg + prop_recall_avg == 0:
            return 0

        return 2 * (prop_precision_avg * prop_recall_avg) / (prop_precision_avg + prop_recall_avg), prop_precision_avg, prop_recall_avg


    def compute_propensities(self, y_true):
        """
        Метод для вычисления пропенсити-фактора для каждой метки.
        """
        label_counts = Counter(list(chain(*y_true)))
        C = (np.log2(len(y_true)) - 1) * (self.B + 1) ** self.A

        propensities = {}
        for label in self.labels:
            count = label_counts[label]
            propensities[label] = 1 / (1 + C * np.exp(-self.A * np.log2(count + self.B)))
        return propensities




