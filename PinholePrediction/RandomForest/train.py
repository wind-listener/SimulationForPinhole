import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # 用于模型保存
from datetime import datetime

# 加载数据
file_path = r"E:\WPSSync\Projects\PinholeAnalysis\PinholePrediction\RandomForest\Dataset\annotation3.csv"
data = pd.read_csv(file_path)

# 预处理数据
X = data.drop(["ImageID", "Diameter"], axis=1)
y = data["Diameter"]
# 获取特征名称
feature_names = X.columns.tolist()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建随机森林回归模型
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=42)

# 训练模型
rf_regressor.fit(X_train, y_train)

# 打印训练结果
train_score = rf_regressor.score(X_train, y_train)
test_score = rf_regressor.score(X_test, y_test)
print(f"训练集得分: {train_score:.4f}")
print(f"测试集得分: {test_score:.4f}")

# 保存模型
# 获取当前日期和时间作为文件名的一部分
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 构建带有时间戳的文件名
model_filename = f"models/rf_regressor_model_{current_time}.pkl"
joblib.dump(rf_regressor, model_filename)
print("model has been saved to ", model_filename)
