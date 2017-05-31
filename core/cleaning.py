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


def first_iteration(data):
    delete_column(data, ['culture_objects_top_25_raion','railroad_terminal_raion','build_count_foam', 'railroad_1line', 'trc_sqm_500', "cafe_count_500_price_high", "mosque_count_500", "leisure_count_500",
                         "office_sqm_1000", "trc_sqm_1000", "cafe_count_1000_price_high", "mosque_count_1000",
                         "cafe_count_1500_price_high", "mosque_count_1500", "cafe_count_2000_price_high"])




if __name__ == '__main__':
    data = utils.load_data('../files/train.csv')

    first_iteration(data)

    # print(data['price_doc'].describe())
    print(data.head())
    utils.save_data(data, 'clean_train.csv')
    # graph_outliers(data)
