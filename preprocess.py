from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# 데이터 로드
data = pd.read_csv(r'C:\Users\khyj\gwp_tabular\heart_failure_balanced_5_5.csv')

# 결측값 확인
print(data.isnull().sum())

# 특성과 타겟 변수 분리
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# 수치형과 범주형 변수 구분
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# 전처리 파이프라인 구성
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 결측값 평균으로 대체
    ('scaler', StandardScaler())                 # 스케일링
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측값 최빈값으로 대체
    ('onehot', OneHotEncoder(drop='first'))               # 원-핫 인코딩
])

# 컬럼 변환기 구성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 데이터 전처리
X_processed = preprocessor.fit_transform(X)

# 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)


X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)