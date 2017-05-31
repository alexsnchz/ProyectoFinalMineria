import pandas
import numpy


def open_file(file_path):
    data = pandas.read_csv(file_path)

    return data


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


def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:, i]
        dict = numpy.unique(numpy_data[:, i])
        for j in range(len(dict)):
            temp[numpy.where(numpy_data[:, i] == dict[j])] = j

        numpy_data[:, i] = temp

    return numpy_data


def first_iteration(data):
    delete_column(data, '')


if __name__ == '__main__':
    data = open_file('../files/train.csv')

    first_iteration(data)

    print(data['Electrical'].mode())
    # print(data[:10])
