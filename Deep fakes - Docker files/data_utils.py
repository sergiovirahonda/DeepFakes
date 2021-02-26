import datetime
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import zipfile
import cv2
import sys


def load_data(args):
    sys.stdout.write("Starting data processing")
    
    print('Current directory:', flush=True)
    print(os.getcwdb(), flush=True)
    entries = os.listdir('/root/DeepFakesDataset')
    print(entries)
    print('Extracting files', flush=True)

    file_1 = '/root/DeepFakesDataset/trump.zip'
    file_2 = '/root/DeepFakesDataset/biden.zip'
    extract_to = '/root/DeepFakesDataset/'

    with zipfile.ZipFile(file_1, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    with zipfile.ZipFile(file_2, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print('Files extracted', flush=True)

    faces_1 = create_dataset('/root/DeepFakesDataset/trump')
    print('First dataset created', flush=True)
    faces_2 = create_dataset('/root/DeepFakesDataset/biden')
    print('Second dataset created', flush=True)

    print("Total President Trump face's samples: ",len(faces_1))
    print("Total President Biden face's samples: ",len(faces_2))

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(faces_1, faces_1, test_size=0.20, random_state=0)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(faces_2, faces_2, test_size=0.15, random_state=0)

    print(X_train_a.shape)
    print(X_train_a[0])


    return X_train_a, X_test_a, X_train_b, X_test_b


def create_dataset(path):
    images = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            image = cv2.imread(os.path.join(dirname, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32')
            image /= 255.0
            images.append(image)
    images = np.array(images)
    return images


def save_model(bucket_name, best_model):
    print('inside save_model, about to save it to GCP Storage',flush=True)
    bucket = storage.Client().bucket(bucket_name)
    print(bucket)
    blob1 = bucket.blob('{}/{}'.format(datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S'),best_model))
    blob1.upload_from_filename(best_model)
