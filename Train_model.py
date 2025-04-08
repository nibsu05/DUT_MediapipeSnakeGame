# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu
data = pd.read_csv('hand_gesture_data.csv', header=None)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thiết lập mô hình Random Forest
clf = RandomForestClassifier(random_state=42)

# Tạo grid các tham số để thử
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearch để tìm bộ tham số tốt nhất (dùng 5-fold cross-validation)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1)

# Train
grid_search.fit(X_train, y_train)

# In kết quả
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Đánh giá độ chính xác
y_pred = best_model.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))

# Lưu model
joblib.dump(best_model, 'hand_gesture_model.pkl')
print("Đã lưu model tốt nhất vào hand_gesture_model.pkl")
