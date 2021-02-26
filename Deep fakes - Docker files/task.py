import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import data_utils
import model
import sys
import os
from google.cloud import storage
import datetime


def train_model(args):

    X_train_a, X_test_a, X_train_b, X_test_b = data_utils.load_data(args)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    # If you're going to use MultiGPU - Create a MirroredStrategy 
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    #     autoencoder_a = model.autoencoder_model()
    #     autoencoder_b = model.autoencoder_model()

    # If you're going to use MultiGPU - Comment out the next two lines
    autoencoder_a = model.autoencoder_model()
    autoencoder_b = model.autoencoder_model()

    print('Starting autoencoder_a training', flush=True)
    checkpoint_1 = ModelCheckpoint("autoencoder_a.hdf5", monitor='val_loss', verbose=1,save_best_only=True, mode='auto', save_freq="epoch")
    autoencoder_a.fit(X_train_a, X_train_a,epochs=args.epochs,batch_size=128,shuffle=True,validation_data=(X_test_a, X_test_a),callbacks=[checkpoint_1])
    print('Training A has ended. ',flush=True)

    print('Starting autoencoder_b training', flush=True)
    checkpoint_2 = ModelCheckpoint("autoencoder_b.hdf5", monitor='val_loss', verbose=1,save_best_only=True, mode='auto', save_freq='epoch')
    autoencoder_b.fit(X_train_a, X_train_a,epochs=args.epochs,batch_size=128,shuffle=True,validation_data=(X_test_b, X_test_b),callbacks=[checkpoint_2])
    print('Training B has ended. ',flush=True)

    print('Proceeding to save models into GCP.',flush=True)

    data_utils.save_model(args.bucket_name,'autoencoder_a.hdf5')
    print('autoencoder_a saved at GCP Storage', flush=True)
    data_utils.save_model(args.bucket_name,'autoencoder_b.hdf5')
    print('autoencoder_b saved at GCP Storage', flush=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name',
                        type=str,
                        help='GCP bucket name')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Epochs number')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    train_model(args)

if __name__ == '__main__':
    main()