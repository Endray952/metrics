from sklearn.metrics import f1_score
from collections import defaultdict

# Пример истинных и предсказанных меток в виде множеств
y_true = [
    {'suggestion', 'general_error', 'content_download_error'},
    {'general_error'},
    {'suggestion', 'network_issues'},
    {'advertisement_complaints', 'price_policy_complaints'}
]

y_pred = [
    {'suggestion', 'content_download_error'},
    {'general_error', 'network_issues'},
    {'suggestion'},
    {'advertisement_complaints'}
]

# Иерархическая структура меток
label_hierarchy = {
    'general_error': [],
    'general_functionality_or_design_issues': ['general_error'],
    'network_issues': ['general_functionality_or_design_issues'],
    'content_download_error': ['general_functionality_or_design_issues'],
    'app_work_issues': ['general_functionality_or_design_issues'],
    'crediting_problems': ['general_functionality_or_design_issues'],
    'advertisement_complaints': ['general_error'],
    'price_policy_complaints': ['general_error'],
    'content_issues': ['general_error'],
    'content_doesnt_work': ['general_error'],
    'suggestion': [],
    'question': []
}

def get_ancestors(label, hierarchy):
    """Возвращает множество всех предков для данной метки."""
    ancestors = set()
    stack = [label]
    while stack:
        current = stack.pop()
        parents = hierarchy.get(current, [])
        ancestors.update(parents)
        stack.extend(parents)
    return ancestors

def compute_hierarchical_f1(y_true, y_pred, hierarchy):
    """Вычисляет иерархический F1-Score."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_labels, pred_labels in zip(y_true, y_pred):
        true_ancestors = set()
        pred_ancestors = set()

        for label in true_labels:
            true_ancestors.update(get_ancestors(label, hierarchy))
            true_ancestors.add(label)

        for label in pred_labels:
            pred_ancestors.update(get_ancestors(label, hierarchy))
            pred_ancestors.add(label)

        true_positives += len(true_ancestors & pred_ancestors)
        false_positives += len(pred_ancestors - true_ancestors)
        false_negatives += len(true_ancestors - pred_ancestors)

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

# Вычисление иерархического F1-Score
hierarchical_f1 = compute_hierarchical_f1(y_true, y_pred, label_hierarchy)
print(f"Set Based Hierarchical F1-Score: {hierarchical_f1:.2f}")