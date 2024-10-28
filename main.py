import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from IPython.display import Image, display

# Colab에서 그래프가 잘 보이도록 설정
%matplotlib inline

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# List of models
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Extra Trees": ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

# Save results in a list
results = []

# Training and evaluating each model
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predictions
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save the results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Display results in a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plotting AUC ROC Curve for each model
plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]  # Get probability for the positive class
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

# Graph settings
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve for Different Models')
plt.legend(loc='lower right')

# Save figure and display
plt.savefig("roc_curve.png")  # 파일로 저장
display(Image(filename="roc_curve.png"))  # 이미지 표시
