import numpy as np


# Random for problem categories
def rand_cat(data: np.ndarray, label_cat: list, num_data: int, value_seed: int):
    """

    :param data:
    :param label_cat:
    :param num_data:
    :param value_seed:
    :return:
    """
    cat = []
    np.random.seed(value_seed)
    dt = np.empty((data.shape[0]*num_data, data.shape[1]))
    for i in range(num_data):
        random = (1/10)*np.random.rand(data.shape[0], data.shape[1])
        _ = data + random
        dt[data.shape[0]*i:data.shape[0]*(i+1), :] = _
        cat.append(label_cat)

    cat = np.array(cat).reshape(-1, 1)
    return dt, cat


def rand_regression(data: np.ndarray,
                    label_regress: list,
                    num_data: int,
                    value_seed: int,
                    category_FORMATS: list):
    """
    :param data:
    :param label_regress:
    :param num_data:
    :param value_seed:
    :param category_FORMATS:
    :return:
    """
    np.random.seed(value_seed)
    label = []
    # Adding label
    for i in range(len(label_regress)):
        for j in range(len(label_regress[0])):
            for z in range(num_data):
                label.append(label_regress[i][j])
    label = np.array(label).reshape(-1, 1)
    # random value in label regression
    label += np.random.rand(label.shape[0], label.shape[1])
    dt = np.empty((len(label_regress[0]), data.shape[1]))
    # Random data
    for i in range(data.shape[0]):
        for j in range(num_data):
            random = (1/10)*np.random.rand(1, data.shape[1])
            if i == j == 0:
                dt = data[i, :] + random
            else:
                _ = data[i, :] + random
                dt = np.vstack((dt, _))

    return dt, label


# Thresh hold data
def threshold_data(df: dict, n_data: int, category: int, threshold: float):
    """

    :param category:
    :param df: Dataframe of dataset.
    :param n_data: Amount data with specific predict category in model_cls.
    :param threshold:
    :return: Data clear -> Type: np.ndarray

        Use data in xlsx not clear.
        We can filter value not necessary to remove.
    """
    # Start V(gas)/V(air) ~ 1 or equal 1.
    start_value = 1
    number_random = 50
    data = np.empty((n_data * category, category))
    for i, index_df in enumerate(df):
        flag = 0
        dt = df[index_df].to_numpy()[:, 1:]
        for index_data in range(dt.shape[0]):
            for i_bool in dt[index_data, :] > (start_value + threshold):
                if i_bool and flag == 0:
                    flag = 1
                    data[n_data * i:(i + 1) * n_data, :] = dt[index_data:index_data + n_data, :]

    return data.reshape((category, category * n_data))


# Convert number: int to label : string
def convert_num_string(category: list, pred):
    """

    :param category: [Class]: Example: ['NH3' , 'H2S', 'Methanol' ,.....]
    :param pred: Prob predict about class ? base on value max at index.
    :return: type string.
    """

    assert isinstance(pred, (np.ndarray, list)), f"Parameter support for " \
                                                 f"two type is {np} and {list}"

    # type list
    if isinstance(pred, list):
        return category[pred.index(max(pred))]
    # type numpy
    if isinstance(pred, np.ndarray):
        return category[np.where(pred == np.amax(pred))[0][0]]
