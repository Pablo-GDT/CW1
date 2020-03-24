from os import path
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


# def view_data_segments(xs, ys):
#     """Visualises the input file with each segment plotted in a different colour.
#     Args:
#         xs : List/array-like of x co-ordinates.
#         ys : List/array-like of y co-ordinates.
#     Returns:
#         None
#     """
#     assert len(xs) == len(ys)
#     assert len(xs) % 20 == 0
#     len_data = len(xs)
#     num_segments = len_data // 20
#     colour = np.concatenate([[i] * 20 for i in range(num_segments)])
#     plt.set_cmap('Dark2')
#     plt.scatter(xs, ys, c=colour)
#     plt.show()


def segment_data(xs, ys):
    """Splits the data into segments of 20 points
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        x_segments: list/array of x segments
        y_segments: list/array of y segments"""
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
    """Estimates the values of the weights, gradient(m) and y-intercept(c) for a linear model y= ax + c for the given data
    Args
        xs : List/array-like of 20 x co-ordinates.
        ys : List/array-like of 20 y co-ordinates.
    Returns:
        y_hat:  list of estimated values using linear least squares
        linear_err: total least squares error for linear model on given data"""
    ones = np.ones(xs.shape)
    X_array = np.column_stack((ones, xs))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    linear_weights = x_e.dot(X_array.T).dot(ys)

    c, m = linear_weights[0], linear_weights[1]
    y_hat = xs*m + c
    linear_err = sum_squared(y_hat, ys)
    print("linear error =" + str(linear_err))
    return y_hat, linear_err


def sum_squared(y, y_hat):
    """Calculates the sum squared error of the residuals between the y values for the modelled points and the data
    Args
        y_hat : List/array-like of modelled y_values
        y : List/array-like of y values from data
    Returns:
        sum squared error: the sum of the residuals between estimated y-values and y-values of data"""
    return np.sum((y_hat-y)**2)


def polynomial_least_squares(xs, ys):
    """Estimates the values of the coefficients a and c for the linear model y= ax^3 bx^2 + cx +d
    Args
        x : List/array-like of x co-ordinates.
        y : List/array-like of y co-ordinates.
    Returns:
        polynomial_weights:  list of polynomial coefficients"""

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
    """Estimates the weights of a cubic model  a, b, c and d for the cubic model y= ax^3 bx^2 + cx + d for the given data points
    Args
        x : List/array-like of x co-ordinates.
        y : List/array-like of y co-ordinates.
    Returns:
         y_hat:  list of estimated values using cubic model
        cubic_err: total least squares error for cubic model on given data"""

    ones = np.ones(xs.shape)
    xs2 = np.square(xs)
    xs3 = np.power(xs, 3)

    X_array = np.column_stack((ones, xs, xs2, xs3))
    x_e = np.linalg.inv(X_array.T.dot(X_array))
    cubic_weights = x_e.dot(X_array.T).dot(ys)

    y_hat = xs3 * cubic_weights[3] + xs2 * \
        cubic_weights[2] + xs * cubic_weights[1] + cubic_weights[0]
    cubic_err = sum_squared(y_hat, ys)
    print("cubic error =" + str(cubic_err))
    return y_hat, cubic_err


def sinusoidal_least_squared(xs, ys):
    """Estimates the values of the coefficients for a and v for the sinusoidal model y= a(sin(x)) + b for the given data points
    Args
        x : List/array-like of 20 x co-ordinates.
        y : List/array-like of 20 y co-ordinates.
    Returns:
         y_hat:  list of estimated values using linear least squares
        sin_err: total least squares error for sinusoidal model on given data"""
    ones = np.ones(xs.shape)
    X = np.column_stack((ones, np.sin(xs)))
    sin_weights = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(ys)
    # sin equation:
    y_hat = sin_weights[1] * np.sin(xs) + sin_weights[0]

    sin_err = sum_squared(y_hat, ys)
    print("sin error =" + str(sin_err))
    return y_hat, sin_err


def quartic_least_squares(xs, ys):
    """Estimates the values of the coefficients a and c for the linear model y= ax^3 bx^2 + cx +d
    Args
        x : List/array-like of x co-ordinates.
        y : List/array-like of y co-ordinates.
    Returns:
        polynomial_weights:  list of polynomial coefficients"""
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


def fitting_best_functions(x_segments, y_segments):
    """Find the best fitting functions out of linear, cubic and sinusoidal functions and creates a list of their y_values
    Args:
        x_segments : List/array-like of 20 x co-ordinates.
        y_segments : List/array-like of 20 y co-ordinates.
    Returns:
        best_y_hat: best y co-ordiantes given the best fitting function
        total_reconstructed_error: total least squared error for best fitting function"""
    total_reconstructed_error = 0
    best_y_hat = []
    # looping through the segments
    for xs, ys in zip(x_segments, y_segments):

        # model signal using linear least squares method and calculate the error for each function
        y_hat_linear, linear_err = linear_least_squares(xs, ys)

        y_hat_poly, polynomial_err = polynomial_least_squares(xs, ys)

        y_hat_cubic, cubic_err = cubic_least_squares(xs, ys)

        y_hat_quartic, quartic_err = quartic_least_squares(xs, ys)

        y_hat_sin, sin_err = sinusoidal_least_squared(xs, ys)
        # finds the smallest error aa
        best_fit = min(linear_err, cubic_err, sin_err)
        total_reconstructed_error = total_reconstructed_error + best_fit
        # check which function produced the smallest error and add it the best fitting points
        if best_fit == linear_err:
            best_y_hat.extend(y_hat_linear)
        elif best_fit == cubic_err:
            best_y_hat.extend(y_hat_cubic)
        else:
            best_y_hat.extend(y_hat_sin)
    return best_y_hat, total_reconstructed_error


def plot_graph(x_coordiantes, y_coordiantes, best_y_hat):
    """Plots a graph of the reconstructed functions onto the x and y coordinates
    Args
        x_coordiantes : List/c array-like of x co-ordinates.
        y_coordiantes: List/array-like of y co-ordinates.
        best_y_hat: The best fitting y values
    Returns: None"""

    fig, axs = plt.subplots()
    axs.scatter(x_coordiantes, y_coordiantes, label="Data")
    plt.plot(x_coordiantes, best_y_hat, label="Fitted functions", color="red")
    plt.xlabel("x-axis", fontsize=12)
    plt.ylabel("y-axis", fontsize=12)
    plt.title("Fitting function to datapoints", fontsize=12)
    fig.legend()
    plt.show()


def main():
    # reads in command line arguments passed into the script and stores them. It then carries out validity checks making sure there's at least a file argument and that the second argument if given is '--plot'
    sys_arguments = sys.argv[1:]
    if len(sys_arguments) == 0:
        sys.exit("Error: You need to pass in a valid csv file")
    elif len(sys_arguments) > 2:
        sys.exit(
            "Error: Too many arguments; Program only accepts a valid csv file and an optional '--plot' argument")
    elif len(sys_arguments) == 2 and sys_arguments[1] != '--plot':
        sys.exit(
            "Error: Unkown second argument; program only accepts  '--plot' as an optional second argument")
    else:
        # grabs the name of the csv file and check if it's a valid file
        csvfile_name = sys_arguments[0]
        try:
            path.isfile(csvfile_name)
        except FileNotFoundError:
            print("The filename given is not recognised. Please give a valid csv file")
        # reads and separates points from csv file into x-coordiantes and y-coordiantes and then segments the coordinates into individual signals
        x_coordiantes, y_coordiantes = load_points_from_file(csvfile_name)
        x_segments, y_segments = segment_data(x_coordiantes, y_coordiantes)
        # fits the best function to each signal and returns the coordinates and reconstruction error which is printed
        best_y_hat, total_reconstructed_error = fitting_best_functions(x_segments, y_segments)
        print(total_reconstructed_error)
        # logical statement that identifies when the user passes the plotting argument
        if len(sys_arguments) == 2 and sys_arguments[1] == '--plot':
            plot_graph(x_coordiantes, y_coordiantes, best_y_hat)


if __name__ == '__main__':
    main()
