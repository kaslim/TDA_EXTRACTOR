import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def read_variable_length_csv(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_data = line.strip().split(',')
            data_list.append(line_data)
    max_features = max(len(line_data) for line_data in data_list) - 2  # Subtract SampleName and Label
    column_names = ['SampleName', 'Label'] + [f'Feature{i}' for i in range(1, max_features + 1)]
    df = pd.DataFrame(data_list, columns=column_names)
    # Randomly shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# Path to your CSV file
csv_file_path = '/Users/yuxuanliu/PycharmProjects/TDA-extractor/training-b.csv'
data = read_variable_length_csv(csv_file_path)

# Replace 'None' with zeros and convert features to floats
features = data.iloc[:, 2:].fillna(0).astype(float).values

# Encode the labels
labels = LabelEncoder().fit_transform(data.iloc[:, 1].values)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=43)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=43)
xgb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of XGBoost model: {accuracy:.4f}')