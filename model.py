# model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터 로드 (전처리된 데이터 사용)
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# 모델 리스트
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Extra Trees": ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

# 모델별 학습 및 평가 결과 저장 함수
def train_and_evaluate():
    results = []
    
    # 모델별 학습 및 평가
    for model_name, model in models.items():
        model.fit(X_train, y_train)  # 모델 학습
        y_pred = model.predict(X_test)  # 예측

        # 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 결과 저장
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Model_Object": model  # 모델 객체도 저장
        })

    # 결과를 데이터프레임으로 반환
    results_df = pd.DataFrame(results)
    return results_df
