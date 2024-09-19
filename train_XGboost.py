import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import wandb
from wandb.integration.xgboost import WandbCallback

# 控制是否使用wandb
use_wandb = True

if use_wandb:
    # 初始化wandb
    wandb.init(project="TDA-extractor", config={
        "num_round": 100,
        "chunk_size": 1000,
        "random_state": 43,
        "objective": "multi:softmax",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "predictor": "cpu_predictor"
    })
    config = wandb.config
else:
    # 使用默认配置
    config = {
        "num_round": 100,
        "chunk_size": 1000,
        "random_state": 43,
        "objective": "multi:softmax",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "predictor": "cpu_predictor"
    }


def load_features(file_paths):
    features, labels = [], []
    max_length = 0
    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)
        for item in data:
            if len(item) < 3:
                print(f"Skipping item with insufficient length: {item}")
                continue
            feature_values = item[2:]
            features.append(feature_values)
            labels.append(item[1])
            max_length = max(max_length, len(feature_values))
    return features, labels, max_length


def parse_features(features, max_length):
    parsed_features = []
    for item in features:
        feature_values = [float(val.split(', ')[1].strip(']')) for val in item]
        if len(feature_values) < max_length:
            feature_values.extend([0] * (max_length - len(feature_values)))
        parsed_features.append(feature_values)
    return np.array(parsed_features)


# 读取和解析训练数据
train_file_paths = ['training-a_features.npy', 'training-b_features.npy', 'training-c_features.npy', 'validation_features.npy',
                    'training-d_features.npy', 'training-e_features.npy', 'training-f_features.npy']
val_file_paths = ['validation_features.npy']

train_features, train_labels, max_train_length = load_features(train_file_paths)
val_features, val_labels, max_val_length = load_features(val_file_paths)

print(f"Loaded {len(train_features)} training features and {len(train_labels)} training labels")
print(f"Loaded {len(val_features)} validation features and {len(val_labels)} validation labels")

max_length = max(max_train_length, max_val_length)

train_features_parsed = parse_features(train_features, max_length)
val_features_parsed = parse_features(val_features, max_length)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_parsed)
val_features_scaled = scaler.transform(val_features_parsed)

X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, train_labels_encoded,
                                                  test_size=0.2, random_state=config["random_state"])

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 设置参数
params = {
    'objective': 'multi:softmax',
    'num_class': config["num_class"],
    'eval_metric': config["eval_metric"],
    'tree_method': config["tree_method"],
    'random_state': config["random_state"]
}


# 自定义评估函数来计算准确率和F1 Score
def custom_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.array(preds)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return [('accuracy', accuracy), ('f1', f1)]


# 回调函数用于在每个epoch后计算和记录准确率和F1 Score
class CustomCallback(xgb.callback.TrainingCallback):
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb

    def after_iteration(self, model, epoch, evals_log):
        preds = model.predict(dval)
        preds = np.array(preds)
        accuracy = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')

        if self.use_wandb:
            wandb.log({"epoch": epoch, "accuracy": accuracy, "f1": f1})

        print(f"Epoch {epoch}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")
        return False


# 训练模型
if use_wandb:
    bst = xgb.train(params, dtrain, num_boost_round=config["num_round"], evals=[(dval, 'eval')],
                    callbacks=[CustomCallback(use_wandb)])
else:
    bst = xgb.train(params, dtrain, num_boost_round=config["num_round"], evals=[(dval, 'eval')],
                    callbacks=[CustomCallback(use_wandb)])

# 验证模型
dval_full = xgb.DMatrix(val_features_scaled, label=val_labels_encoded)
val_predictions = bst.predict(dval_full)
val_predictions_labels = np.array(val_predictions)
val_accuracy = accuracy_score(val_labels_encoded, val_predictions_labels)
val_f1 = f1_score(val_labels_encoded, val_predictions_labels, average='macro')

print(f'Validation Accuracy of XGBoost model: {val_accuracy:.4f}')
print(f'Validation F1 Score of XGBoost model: {val_f1:.4f}')

if use_wandb:
    wandb.log({"Validation Accuracy": val_accuracy, "Validation F1 Score": val_f1})
