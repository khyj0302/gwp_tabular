# main.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from model import train_and_evaluate  # model.py에서 함수 불러오기

# 데이터 로드 (전처리된 데이터 사용)
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# 모델 학습 및 평가
results_df = train_and_evaluate()
print(results_df)

# AUC ROC 커브 시각화
plt.figure(figsize=(10, 8))

for index, row in results_df.iterrows():
    model_name = row['Model']
    model = row['Model_Object']
    y_proba = model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 예측 확률
    
    # ROC 커브 계산
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

# 그래프 설정
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve for Different Models')
plt.legend(loc='lower right')
plt.show()
