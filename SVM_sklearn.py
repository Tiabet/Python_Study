from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 데이터셋 로드 (Iris 데이터셋)
iris = datasets.load_iris()
X = iris.data  # 특징 값 (꽃잎, 꽃받침 길이 및 너비)
y = iris.target  # 클래스 레이블

# 이진 분류를 위해 클래스 0과 1만 사용
X = X[y != 2]
y = y[y != 2]

# 데이터 분리 (학습용과 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 모델 생성 및 학습
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# 예측 수행
y_pred = svm_model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 상세 리포트 출력
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
