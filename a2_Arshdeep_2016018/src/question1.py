''' Refernces
https://homes.cs.washington.edu/~thickstn/spectrograms.html
https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.io import wavfile
import os
from scipy.signal import get_window
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import pickle
from random import randint as ri


def discrete_fourier_transform(arr):

    N = arr.shape[0]
    dft_array = [0 for i in range(N)]
    i_array = np.arange(0, N, 1)
    j_array = np.arange(0, N, 1)

    for i in range(N):
        dft_array[i] = np.dot(np.exp(-2j * np.pi * i_array[i] * j_array/N), arr)

    dft_array = np.array(dft_array)
    return dft_array


eps = 1e-14

if __name__ == "__main__":

    fs = 16000
    file_result = open('results.txt', 'a')
    version_count = 29
    for window_size_value in [15]:
        for stride_value in [75]:
            for kernel in ['linear']:
                for impurity in [0]:
                    print(version_count, window_size_value, stride_value, kernel, impurity)
                    file_result.write(str(version_count) + "\n")
                    file_result.write("Window Size : " + str(window_size_value) + " ")
                    file_result.write("Stride Value : " + str(stride_value) + " ")
                    file_result.write("Kernel : " + kernel + " Noise Percentage : " + str(0) + " ")
                    file_result.write("With Log and Squared Changes with 20 log and With Normalization \n\n")

                    window_size = int(0.001 * window_size_value * fs)
                    stride = int(0.001 * window_size_value * stride_value * 0.01 * fs)
                    print(window_size, stride)

                    train_X = []
                    train_Y = []
                    test_X = []
                    test_Y = []

                    mapping_output = {
                        'zero': 0,
                        'one': 1,
                        'two': 2,
                        'three': 3,
                        'four': 4,
                        'five': 5,
                        'six': 6,
                        'seven': 7,
                        'eight': 8,
                        'nine': 9
                    }

                    noise_files = {
                        1: 'doing_the_dishes.wav',
                        2: 'dude_miaowing.wav',
                        3: 'exercise_bike.wav',
                        4: 'pink_noise.wav',
                        5: 'running_tap.wav',
                        6: 'white_noise.wav'
                    }

                    for directory in os.listdir('training'):
                        if directory != '.DS_Store':
                            print(directory)
                            for idx, file in enumerate(os.listdir('training/' + directory)):

                                if file.find('.wav') != -1:

                                    fs, data = wavfile.read('training/' + directory + '/' + file)

                                    if fs >= len(data):
                                        data = np.concatenate((data, np.zeros(fs - len(data))))
                                        # noise_index = ri(1, 6)
                                        # noise_fs, noise_data = wavfile.read('_background_noise_/'+noise_files[noise_index][:len(data)])
                                        # data_noise = data + noise_data[:len(data)]
                                        # if idx % impurity == 0:
                                        #     noise_index = ri(1, 6)
                                        #     noise_fs, noise_data = wavfile.read('_background_noise_/'+noise_files[noise_index][:len(data)])
                                        #     data += noise_data[:len(data)]
                                    else:
                                        print(arsh)

                                    Xs = np.zeros((int((len(data) - window_size) / stride) + 1, int(window_size / 2) + 1))
                                    Xs_noise = np.zeros((int((len(data) - window_size) / stride) + 1, int(window_size / 2) + 1))
                                    window = get_window("hann", window_size, fftbins=True)

                                    for i in range(Xs.shape[0]):
                                        Xs[i] = 20 * np.log10((np.abs(
                                            discrete_fourier_transform(data[i * stride:i * stride + window_size] * window)[
                                            :int(window_size / 2) + 1]) ** 2) / window_size + eps)

                                        # Xs_noise[i] = 20 * np.log10((np.abs(
                                        #     fft(data_noise[i * stride:i * stride + window_size] * window)[
                                        #     :int(window_size / 2) + 1]) ** 2) / window_size + eps)

                                    train_X.append(Xs)
                                    # train_X.append(Xs_noise)
                                    train_Y.append(mapping_output[directory])
                                    # train_Y.append(mapping_output[directory])

                    train_X = np.array(train_X)
                    train_Y = np.array(train_Y)
                    print(train_X.shape, train_Y.shape)
                    # pickle.dump(train_X, open('Q1_train_X_Submission.pkl', 'wb'))
                    # pickle.dump(train_Y, open('Q1_train_Y_Submission.pkl', 'wb'))

                    for directory in os.listdir('validation'):
                        if directory != '.DS_Store':
                            print(directory)
                            for file in os.listdir('validation/' + directory):

                                if file.find('.wav') != -1:

                                    fs, data = wavfile.read('validation/' + directory + '/' + file)

                                    if fs >= len(data):
                                        data = np.concatenate((data, np.zeros(fs - len(data))))
                                    else:
                                        print(arsh)

                                    wps = fs / float(window_size)
                                    Xs = np.zeros((int((len(data) - window_size) / stride) + 1, int(window_size / 2) + 1))
                                    window = get_window("hann", window_size, fftbins=True)

                                    for i in range(Xs.shape[0]):
                                        Xs[i] = 20 * np.log10((np.abs(
                                            discrete_fourier_transform(data[i * stride:i * stride + window_size] * window)[
                                            :int(window_size / 2) + 1]) ** 2) / window_size + eps)

                                    test_X.append(Xs)
                                    test_Y.append(mapping_output[directory])

                    test_X = np.array(test_X)
                    test_Y = np.array(test_Y)

                    # pickle.dump(test_X, open('Q1_test_X_Submission.pkl', 'wb'))
                    # pickle.dump(test_Y, open('Q1_test_Y_Submission.pkl', 'wb'))

                    print(test_X.shape, test_Y.shape)

                    train_X = train_X.reshape(train_X.shape[0], -1)
                    test_X = test_X.reshape(test_X.shape[0], -1)

                    train_X = normalize(train_X)
                    test_X = normalize(test_X)

                    svm_model_linear = SVC(kernel=kernel, C=1).fit(train_X, train_Y)
                    # svm_model_linear = pickle.load(open('Q1_model_1.pkl', 'rb'))
                    pickle.dump(svm_model_linear, open('Q1_model_' + str(version_count) + '.pkl', 'wb'))

                    svm_predictions_test = svm_model_linear.predict(test_X)

                    report_test = classification_report(test_Y, svm_predictions_test)

                    file_result.write(report_test)
                    file_result.write("\n")
                    print(report_test)
                    print("\n")

                    svm_predictions_train = svm_model_linear.predict(train_X)

                    report_train = classification_report(train_Y, svm_predictions_train)

                    file_result.write(report_train)
                    file_result.write("\n\n\n\n")
                    print(report_train)
                    print("\n\n\n\n")

                    version_count += 1
