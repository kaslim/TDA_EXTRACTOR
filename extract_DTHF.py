import os
import pandas as pd
import numpy as np
from feature_extraction.feature_extractor import extract_features
import logging


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_dir = ''
    training_dirs = ['dataset/test-a', 'dataset/training-a', 'dataset/training-b', 'dataset/training-c', 'dataset/training-d',
                     'dataset/training-e', 'dataset/training-f']
    validation_dir = 'dataset/validation'

    # 读取验证集文件名
    validation_files = set()
    validation_csv_path = os.path.join(dataset_dir, validation_dir, 'REFERENCE.csv')
    try:
        validation_data = pd.read_csv(validation_csv_path, header=None)
        validation_files.update(validation_data[0].values)
    except Exception as e:
        logging.error(f"Error reading validation CSV: {e}")
        return

    # 提取训练集特征
    for training_dir in training_dirs:
        train_csv_path = os.path.join(dataset_dir, training_dir, 'REFERENCE.csv')
        try:
            train_data = pd.read_csv(train_csv_path, header=None)
        except Exception as e:
            logging.error(f"Error reading training CSV {train_csv_path}: {e}")
            continue

        train_features = []
        for _, row in train_data.iterrows():
            file_name, label = row[0], row[1]
            if file_name not in validation_files:
                file_path = os.path.join(training_dir, file_name)
                label_str = 'Abnormal' if label == 1 else 'Normal'
                features = extract_features(file_path, label_str, '.wav')
                if features:
                    train_features.append(features)
                    logging.info(f"Processed file: {file_path}.wav, Label: {label_str}")

        if train_features:
            np.save(f'{training_dir.split("/")[-1]}_filtered_features.npy', np.array(train_features, dtype=object))
            logging.info(f"Saved training features for {training_dir}")

    # 提取验证集特征
    val_features = []
    for _, row in validation_data.iterrows():
        file_name, label = row[0], row[1]
        file_path = os.path.join(validation_dir, file_name)
        label_str = 'Abnormal' if label == 1 else 'Normal'
        features = extract_features(file_path, label_str, '.wav')
        if features:
            val_features.append(features)
            logging.info(f"Processed file: {file_path}.wav, Label: {label_str}")

    if val_features:
        np.save('validation_filtered_features.npy', np.array(val_features, dtype=object))
        logging.info("Saved validation features")

    logging.info("Features have been extracted and saved to .npy files.")


if __name__ == '__main__':
    main()
