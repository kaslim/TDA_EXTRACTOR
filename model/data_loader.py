import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
            for val in feature_values:
                frame_idx = int(val.split(', ')[0].strip('['))
                max_length = max(max_length, frame_idx + 1)
    return features, labels, max_length

def parse_features(features, max_length):
    parsed_features = []
    for item in features:
        feature_vector = [0] * max_length
        for val in item:
            birth, death = map(int, val.strip('[]').split(', '))
            lifetime = death - birth
            feature_vector[birth] = lifetime
        parsed_features.append(feature_vector)
    return np.array(parsed_features)

def prepare_data(train_paths, val_paths, config):
    # Load and parse features
    train_features, train_labels, max_train_length = load_features(train_paths)
    val_features, val_labels, max_val_length = load_features(val_paths)
    max_length = max(max_train_length, max_val_length)

    train_features_parsed = parse_features(train_features, max_length)
    val_features_parsed = parse_features(val_features, max_length)

    # Label encoding and scaling
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_parsed)
    val_features_scaled = scaler.transform(val_features_parsed)

    X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, train_labels_encoded,
                                                      test_size=0.2, random_state=config["random_state"])
    return X_train, X_val, y_train, y_val
