import csv
import os
import numpy as np
import json
import pandas as pd
from xlrd import open_workbook
from collections import Counter

"""
    Information data:
        + 200 Rows and 5 cols => one point data.
        + Since the gas measurement takes place by the amount of data
          to be taken as mentioned by the parameter self.n_data.
        + Could random data or arguments data -> Increase data -> Improve quantity model_cls neural network.
"""

# path_file = <./file.csv> or <./file.txt> ,......
path_file = 'C:/Users/SONY/Downloads/LR/classification_wine/data.xlsx'

# Category of file .xlsx
Category_FORMATS = ['NH3', 'H2S', 'Methanol', 'Acetone', 'Ethanol']


class LOAD_DATA_EXCEL:
    # Init parameters
    def __init__(self, label: list,
                 num_sheet: int,
                 path: str):
        """
        :param label: Contains the properties of the data
        :param label: Category of data
        :param num_sheet: amount of data to retrieve
        """
        self.rows = None
        self.label = label
        self.path = path_file
        self.n_sheet = num_sheet

        self.data_frame = pd.read_excel(self.path, sheet_name=[Category_FORMATS[i_sheet]\
                                                               for i_sheet in range(self.n_sheet)], engine='openpyxl')

        self.check_paras()
        self.info_data()

    # Check parameter in class LOAD_DATA_EXCEL
    def check_paras(self):
        assert len(Category_FORMATS) >= self.n_sheet > 0, f"Amount data should be bigger than 0 and smaller than {len(Category_FORMATS)}. "
        assert os.path.isfile(self.path), f"Not found file {self.path} in Operation system."
        assert all(isinstance(i_label, str) for i_label in self.label), f"Any element not type str."

    # loadData with file .xlsx
    def loadData(self):
        # Load data.
        # 1000 (feature) is : 200(times) multi with 5 (sensor air)
        # ppm : predict gas concentration => Value is deserted or real
        data_np = np.empty((self.rows//200, 1000))
        # Load categories : classification.
        label_category = []
        # 1000 (feature) is : 200(times) multi with 5 (sensor air)
        label_regression = []
        for i, df in enumerate(self.data_frame):
            assert len(self.data_frame[df].to_numpy()) % 200 == 0, f"data in sheet {df} not available."
            samples = len(self.data_frame[df])//200
            dt = self.data_frame[df].to_numpy()[:, :]
            regress = []
            # Check samples is float or int
            assert isinstance(samples, int), f"samples only support {int}."
            for sample in range(samples):
                data_np[i*samples + sample, :] = dt[200*sample:(sample + 1)*200, :5].reshape(1, -1)
                label_category.append(Category_FORMATS.index(df))
                # Use label regression each num_sheet to predict
                regress.append(dt[200*sample, -1])
            label_regression.append(regress)

        del self.data_frame
        return data_np, label_category, label_regression

    # Count rows appear num_sheet in xlsx
    def info_data(self):
        rows = 0
        for df in self.data_frame:
            rows += len(self.data_frame[df])
        self.rows = rows


# Build class READ_FILE_TXT
class LOAD_DATA_TXT:
    pass

