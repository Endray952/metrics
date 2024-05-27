import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

class LabelMetrics:
    def __init__(self, labels):
        """
        Инициализация класса с множеством меток и параметрами A и B.
        """
        self.labels = labels

    def calculate_label_mentrics(self, y_true, y_pred):
        mlb = MultiLabelBinarizer(
            classes=list(self.labels))
        y_true_binary = mlb.fit_transform(y_true)
        y_pred_binary = mlb.transform(y_pred)

        # Список меток
        labels = mlb.classes_

        # Создание списка для хранения результатов
        results = []

        # Расчет precision, recall и f1-score для каждой метки
        for i, label in enumerate(labels):
            precision = precision_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0)
            recall = recall_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0)
            f1 = f1_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0)
            num_true = np.sum(y_true_binary[:, i])
            num_correct = np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 1))
            results.append({'label': label, 'num_true': num_true, 'num_correct': num_correct, 'precision': precision, 'recall': recall, 'f1': f1})

        micro_f1 = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        macro_f1 = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='macro', zero_division=0)

        return results, micro_f1, macro_f1



