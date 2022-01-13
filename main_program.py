from processing_data import *
from train import *
from read_file import *
import tensorflow as tf

# Folder save model_regress => Predict ppm of gas air
path_dir_regress = 'C:/Users/SONY/Downloads/LR/classification_wine/model_regress/'
# Folder save model_cls => Predict classification
path_dir_cls = 'C:/Users/SONY/Downloads/LR/classification_wine/model_cls/model_1.h5'


# Run program
# The procedure for doing the math is:
#       1. Input: 1000 features normalized
#       2. Predict classification
#       3. After predict class => Predict ppm base linear regression.


def run():
    rf = LOAD_DATA_EXCEL(label=Category_FORMATS, num_sheet=5, path=path_file)
    data, cat, regress = rf.loadData()

    # Take any rows in data to evaluate model good or bad??
    test = data[10].reshape(1, -1)
    test = test/50
    model_cls = tf.keras.models.load_model(path_dir_cls)
    # Probility
    probability_model = tf.keras.Sequential([model_cls])
    # Predict class
    predictions = probability_model.predict(test)
    cls = convert_num_string(Category_FORMATS, predictions[0])

    for file in os.listdir(path_dir_regress):
        if cls in file:
            # Load model
            print(cls)
            loaded_model = pickle.load(open(path_dir_regress + 'weight_' + cls + '.sav', 'rb'))
            result = loaded_model.predict(test)
            print(result)


# Main
if __name__ == '__main__':
    run()