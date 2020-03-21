import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    # """Loads 2d points from a csv called filename
    # Args:
    #     filename : Path to .csv file
    # Returns:
    #     (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    # """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    # """Visualises the input file with each segment plotted in a different colour.
    # Args:
    #     xs : List/array-like of x co-ordinates.
    #     ys : List/array-like of y co-ordinates.
    # Returns:
    #     None
    # """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def segment_data(xs, ys):
    # """Splits the data into segments of 20 points
    # Args:
    #     xs : List/array-like of x co-ordinates.
    #     ys : List/array-like of y co-ordinates.
    # Returns:
    #     x_segments: list/array of x segments
    #     y_segments: list/array of y segments"""
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    x_segments = np.split(xs, num_segments)
    # [xs[x:x + 20] for x in range(0, len(xs), 20)]

    y_segments = np.split(ys, num_segments)
    # [ys[y:y + 20] for y in range(0, len(ys), 20)]
    return x_segments, y_segments


def linear_least_squares(xs, ys):
    # """Estimates the values of the weights a and c for the linear model y= ax + c
    # Args
    #     xs : List/array-like of x co-ordinates.
    #     ys : List/array-like of y co-ordinates.
    # Returns:
    #     y_hat:  list of estimated values using linear least squares"""
    ones = np.ones(xs.shape)
    X_array = np.column_stack((ones, xs))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    linear_weights = x_e.dot(X_array.T).dot(ys)

    c, m = linear_weights[0], linear_weights[1]
    y_hat = xs*m + c
    linear_err = sum_squared(y_hat, ys)
    print("linear error =" + str(linear_err))
    return y_hat, linear_err, c, m


def plot_reconstucted_linear(xs, ys, m, c):
    x_min = xs.min()
    x_max = xs.max()
    y_1r = x_min*m + c
    y_2r = x_max*m + c
    plt.plot([x_min, x_max], [y_1r, y_2r])


def sum_squared(y, y_hat):
    # """Calculates the sum squared error of the residuals between the modelled points and the data
    # Args
    #     y_hat : List/array-like of estiamted y_values
    #     y : List/array-like of y values from data
    # Returns:
    #     sum squared error"""
    return np.sum((y_hat-y)**2)


def plot_reconstucted_func(xs, y_hat):
    plt.plot(xs, y_hat)


def polynomial_least_squares(xs, ys):
    # """Estimates the values of the coefficients a and c for the linear model y= ax^3 bx^2 + cx +d
    # Args
    #     x : List/array-like of x co-ordinates.
    #     y : List/array-like of y co-ordinates.
    # Returns:
    #     polynomial_weights:  list of polynomial coefficients"""

    ones = np.ones(xs.shape)
    xs2 = np.square(xs)

    X_array = np.column_stack((ones, xs, xs2))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    polynomial_weights = x_e.dot(X_array.T).dot(ys)

    y_hat = (xs**2) * polynomial_weights[2] + xs * polynomial_weights[1] + polynomial_weights[0]
    polynomial_err = sum_squared(y_hat, ys)
    print("quadratic error =" + str(polynomial_err))
    return y_hat, polynomial_err


def cubic_least_squares(xs, ys):
    # """Estimates the values of the coefficients a and c for the linear model y= ax^3 bx^2 + cx +d
    # Args
    #     x : List/array-like of x co-ordinates.
    #     y : List/array-like of y co-ordinates.
    # Returns:
    #     polynomial_weights:  list of polynomial coefficients"""

    ones = np.ones(xs.shape)
    xs2 = np.square(xs)
    xs3 = np.power(xs, 3)

    X_array = np.column_stack((ones, xs, xs2, xs3))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    polynomial_weights = x_e.dot(X_array.T).dot(ys)

    y_hat = xs3 * polynomial_weights[3] + xs2 * \
        polynomial_weights[2] + xs * polynomial_weights[1] + polynomial_weights[0]
    cubic_err = sum_squared(y_hat, ys)
    print("cubic error =" + str(cubic_err))
    return y_hat, cubic_err


def sinusoidal_least_squared(xs, ys):
    ones = np.ones(xs.shape)
    X = np.column_stack((ones, np.sin(xs)))
    sin_weights = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(ys)
    # sin equation:
    y_hat = sin_weights[1] * np.sin(xs) + sin_weights[0]

    sin_err = sum_squared(y_hat, ys)
    print("sin error =" + str(sin_err))
    return y_hat, sin_err


def quartic_least_squares(xs, ys):
    # """Estimates the values of the coefficients a and c for the linear model y= ax^3 bx^2 + cx +d
    # Args
    #     x : List/array-like of x co-ordinates.
    #     y : List/array-like of y co-ordinates.
    # Returns:
    #     polynomial_weights:  list of polynomial coefficients"""

    ones = np.ones(xs.shape)
    xs2 = np.square(xs)
    xs3 = np.power(xs, 3)
    xs4 = np.power(xs, 4)

    X_array = np.column_stack((ones, xs, xs2, xs3, xs4))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    polynomial_weights = x_e.dot(X_array.T).dot(ys)

    y_hat = xs4 * polynomial_weights[4] + xs3 * polynomial_weights[3] + xs2 * \
        polynomial_weights[2] + xs * polynomial_weights[1] + polynomial_weights[0]
    quartic_err = sum_squared(y_hat, ys)
    print("quartic error =" + str(quartic_err))
    return y_hat, quartic_err


# def sinusoidal_least_squared(xs, ys):
#     """assuming y=Asin(B(x-c))+D"""
#     A = (ys.max()-ys.min())/2
#     D = (ys.max() + ys.min())/2
#     T = abs(ys.index(min(ys)) - ys.index(min(ys)))
#     B = 2*np.pi/T
#     C = 400
#     return A*sin(B*(xs-C)) + D


# weights = np.polyfit(x_segments[i], y_segments[i], 3)
# print(weights)
# y_hat= weights[0]*x_range**3+ weights[1]*x_range**2 + weights[2]*x_range + weights[3]
# polynomial_err = sum_squared(y_segments[i],y_hat)
# print(polynomial_err)
# plt.plot(x_range,y_hat)
# plt.show()

# weights = np.polyfit(x_segments[i], y_segments[i], 2)
# print(weights)
# y_hat=  weights[0]*x_range**2 + weights[1]*x_range + weights[2]
# polynomial_err = sum_squared(y_segments[i],y_hat)
# print(polynomial_err)


def main():
    # reads in command line parameters, stores th and grabs the name of the csv file
    sys_arguments = sys.argv[1:]
    csvfile_name = sys_arguments[0]

    # reads and separates points from csv file into x-coordiantes and y-coordiantes and then segments the coordinates into individual signals
    x_coordiantes, y_coordiantes = load_points_from_file(csvfile_name)
    view_data_segments(x_coordiantes, y_coordiantes)
    x_segments, y_segments = segment_data(x_coordiantes, y_coordiantes)

    error_list = []
    total_reconstructed_error = 0

    # logical statement that identifies when the user passes the plotting argument
    if len(sys_arguments) == 2 and sys_arguments[1] == '--plot':
        # looping through the segments
        for xs, ys in zip(x_segments, y_segments):

            # model signal using linear least squares method and calculate the error for this method
            y_hat, linear_err, c, m = linear_least_squares(xs, ys)
            plot_reconstucted_linear(xs, ys, m, c)

            y_hat, polynomial_err = polynomial_least_squares(xs, ys)
            plot_reconstucted_func(xs, y_hat)

            y_hat, cubic_err = cubic_least_squares(xs, ys)
            plot_reconstucted_func(xs, y_hat)

            y_hat, quartic_err = quartic_least_squares(xs, ys)
            plot_reconstucted_func(xs, y_hat)

            y_hat, sin_err = sinusoidal_least_squared(xs, ys)
            plot_reconstucted_func(xs, y_hat)

            plt.scatter(xs, ys)
            plt.show()

            error_list = [linear_err, polynomial_err, cubic_err, quartic_err, sin_err]
            total_reconstructed_error = total_reconstructed_error + min(error_list)
            print("total reconstructed error so far:" + str(total_reconstructed_error))
    plt.show()
    if len(sys_arguments) == 2 and sys_arguments[1] == '--plot':


if __name__ == '__main__':
    main()
