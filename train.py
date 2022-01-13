import numpy as np
import os
import pickle
from read_file import *
from processing_data import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Dropout, concatenate, Activation, BatchNormalization, GlobalAvgPool2D, LeakyReLU, Input)
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# File pre_train before.
folder_trained_model = 'C:/Users/SONY/Downloads/LR/classification_wine/model_cls'

# path_file = <./file.csv> or <./file.txt> ,......
path_file = 'C:/Users/SONY/Downloads/LR/classification_wine/data.xlsx'

# Category of file .xlsx
Category_FORMATS = ['NH3', 'H2S', 'Methanol', 'Acetone', 'Ethanol']


# Transform data
def transform_data(X_train, X_test, Y_train, Y_test):
    # Convert 0-255 to 0-1.
    # Use softmax with value very high -> Overflow.
    X_train = X_train/50
    X_test = X_test/50
    # Convert label to encoding.
    # define ordinal encoding.
    encoder = OneHotEncoder(sparse=False)

    # transform data
    Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)

    # Print Y_train and Y_test, X_train and X_test.

    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of Y_train: ", Y_train.shape)
    print("Shape of Y_test: ", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


# Build class Model
class NN:
    # Create init
    def __init__(self, classes, layers: int, unit_layer: list):
        self.classes = classes
        self.layers = layers
        self.unit_layer = unit_layer
        self.save_weights = True
        self.type = True
        self.check()

    def check(self):
        """
            Check parameter : self.classes , self.layers and self.unit_layer
            Parameters use train model_cls => If parameter noy true => model_cls is low efficient
        """
        assert isinstance(self.classes, int), f"Not support type {type(self.classes)} with self.classes."
        assert isinstance(self.layers, int), f"Not support type {type(self.layers)} with self.layers."
        assert isinstance(self.unit_layer, list), f"Not support type {type(self.unit_layer)} with self.unit_layer."
        assert all(unit > 0 for unit in self.unit_layer), f"Any value in {self.unit_layer} smaller than 0."
        assert self.layers == len(self.unit_layer), f"Neural Network not work."

    # Create model_cls
    def model(self, input_shape):
        
        # Input
        input_layer = Input(input_shape)
        
        # Middle
        for layer in range(self.layers):
            _layer_ = Dense(self.unit_layer[layer], activation='relu')(input_layer if layer == 0 else _layer_)
            Dropout(0.5)
        
        # Output
        if self.classes > 1:
            outputs = Dense(self.classes, activation='softmax')(_layer_)
        else:
            outputs = Dense(1, activation='sigmoid')(_layer_)
        # Create model_cls.
        model = Model(inputs=input_layer, outputs=outputs)
        # Compile model_cls
        return model
    
    # Compile
    def cmp(self, X_train, Y_train, X_test, Y_test, model) -> object:
        if self.classes > 1:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            run_history = model.fit(X_train, Y_train, batch_size=100, epochs=200, validation_data=(X_test, Y_test),
                                    verbose=2)

        # Check choose save model_cls ?? Yes or no

        model.save('./model_cls/model_' + str(len(os.listdir(folder_trained_model)) + 1) + '.h5')


# Train in classification
def train_cls(data: np.ndarray, label: np.ndarray):
    # Train_split_data.
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, train_size=0.8, random_state=42)
    # Transform data about between [0, 1]
    X_train, X_test, Y_train, Y_test = transform_data(X_train, X_test, Y_train, Y_test)

    # Train
    model_cat = NN(classes=len(Y_train[0]), layers=2, unit_layer=[100, 200])
    model_cat.model(X_train.shape[1:]).summary()
    model_cat.cmp(X_train, Y_train, X_test, Y_test, model_cat.model(X_train.shape[1:]))


def train_regress(data: np.ndarray, label: np.ndarray, category: list):
    # Path_file <path>/<.folder>
    path_dir = 'D:/Machine_Learning/classification_wine/model_regress/'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    sample = data.shape[0]//len(category)
    for i, cat in enumerate(category):
        X, Y = data[sample*i: sample*(i+1), :], label[sample*i:sample*(i+1)]
        X = X/50
        model = LinearRegression()
        model.fit(X, Y)
        # Save the model to disk
        filename = 'weight_' + category[i] + '.sav'
        pickle.dump(model, open(path_dir + filename, 'wb'))


# Run core of program
def main():
    rf = LOAD_DATA_EXCEL(label=Category_FORMATS, num_sheet=5, path=path_file)
    data, cat, regress = rf.loadData()
    # Adding data categories with algorithms random
    data_cat, cat = rand_cat(data, cat, num_data=50, value_seed=42)
    # Adding data regression with algorithms random
    data_regress, regress = rand_regression(data, regress, num_data=50, value_seed=42, category_FORMATS=Category_FORMATS)

    # Train classification
    train_cls(data_cat, cat)
    # Train regression
    train_regress(data=data_regress, label=regress, category=Category_FORMATS)


if __name__ == '__main__':
    main()
