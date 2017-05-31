from utils import utils
from data_preprocessing import dimensionality_reduction
import matplotlib.pyplot as plt


def delete_column(data, column):
    data = data.drop(column, axis=1, inplace=True)

    return data


def replace_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp

    return data


def replace_missing_values_with_mean(data, column):
    mean = round(data[column].mean(), 2)
    temp = data[column].fillna(mean)
    data[column] = temp

    return data


def replace_missing_values_with_mode(data, column):
    mode = data[column].mode()
    temp = data[column].fillna(mode.iloc[0])
    data[column] = temp

    return data


def graph_outliers(data):
    plt.boxplot(data['price_doc'])
    plt.show()


def count_na_values(data):
    print(data.isnull().sum(axis=0).reset_index())


def first_iteration(data):
    delete_column(data, ['culture_objects_top_25_raion', 'railroad_terminal_raion', 'build_count_foam', 'railroad_1line',
                         'trc_sqm_500', "cafe_count_500_price_high", "mosque_count_500", "leisure_count_500",
                         "office_sqm_1000", "trc_sqm_1000", "cafe_count_1000_price_high", "mosque_count_1000",
                         "cafe_count_1500_price_high", "mosque_count_1500", "cafe_count_2000_price_high", 'hospital_beds_raion'])
    data = data.fillna(-1)

    data = utils.convert_data_to_numeric(data)
    data = dimensionality_reduction.principal_components_analysis(6, data)

    return data


if __name__ == '__main__':
    train_data = utils.load_data('../files/train.csv')
    test_data = utils.load_data('../files/test.csv')

    print('====================[TRAIN DATA]====================')
    train_data = first_iteration(train_data)
    count_na_values(train_data)
    print(train_data.describe())

    print('====================[TEST DATA]====================')
    test_data = first_iteration(test_data)
    count_na_values(test_data)
    print(test_data.describe())

    # print(train_data.head())
    utils.save_data(train_data, 'clean_train.csv')
    utils.save_data(test_data, 'clean_test.csv')
    # graph_outliers(data)
