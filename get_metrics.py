
import pandas as pd
import os
from label_metrics import LabelMetrics
from propensity_f import PropensityF

# Define label hierarchy
label_hierarchy = {
    1: None,  # suggestion
    2: None,  # general_error
    3: 2,  # general_functionality_or_design_issues
    4: 3,  # network_issues
    5: 3,  # content_download_error
    6: 3,  # app_work_issues
    7: 3,  # crediting_problems
    8: 2,  # advertisement_complaints
    9: 2,  # price_policy_complaints
    10: 2,  # content_issues
    11: 2,  # content_doesnt_work
    12: None  # question
}

label_names = {
    1: "suggestion",
    2: "general_error",
    3: "general_functionality_or_design_issues",
    4: "network_issues",
    5: "content_download_error",
    6: "app_work_issues",
    7: "crediting_problems",
    8: "advertisement_complaints",
    9: "price_policy_complaints",
    10: "content_issues",
    11: "content_doesnt_work",
    12: "question"
}


def get_metrics(y_true, y_pred, folder_name='metrics', prompt=''):
    labels = set(label_names.keys())
    propensity_f, propensity_precision, propensity_recall = PropensityF(labels).calculate_prop_f(y_true, y_pred)
    label_metrics, micro, macro, report = LabelMetrics(labels).calculate_label_mentrics(y_true, y_pred)

    label_metrics_df = pd.DataFrame([metrics.values() for metrics in label_metrics], columns=['label', 'total_count', 'hits', 'precision', 'recall', 'F1-Score'])
    other_metrics_df = pd.DataFrame(
        [
            ['propensity', propensity_f, propensity_precision, propensity_recall],
            ['micro', micro[0], micro[1], micro[2]],
            ['macro', macro[0], macro[1], macro[2]],
        ],
        columns=['metric', 'precision', 'recall', 'f1']
    )
    print(report)


    os.makedirs(folder_name, exist_ok=True)
    pd.DataFrame(report).transpose().to_excel(f"{folder_name}/report.xlsx", index=True)
    label_metrics_df.to_excel(f"{folder_name}/label_metrics.xlsx", index=False)
    other_metrics_df.to_excel( f"{folder_name}/other_metrics_df.xlsx", index=False)

    with open(f"{folder_name}/prompt.txt", 'w') as file:
        # Запись строки в файл
        file.write(prompt)

y_true = [{1, 2}, {3,5,6}, {2}]
y_pred = [{1, 2}, {3,5,6}, {3}]

get_metrics(y_true, y_pred, 'test')
