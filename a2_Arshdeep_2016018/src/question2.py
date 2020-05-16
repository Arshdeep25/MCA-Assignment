'''
References - 
https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1
'''
from scipy import fft
from scipy.fftpack import dct
from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import get_window
import pickle
from random import randint as ri

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize

eps = 1e-14


def freq_to_mel(freq):
    return 1125.0 * np.log(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (np.exp(mels/1125.0) - 1)


if __name__ == "__main__":

    fs = 16000
    file_result = open('results_2.txt', 'a')
    version_count = 8
    for window_size_value in [15]:
        for stride_value in [75]:
            for kernel in ['linear']:
                for impurity in [0]:
                    print(version_count, window_size_value, stride_value, kernel, impurity)
                    file_result.write(str(version_count) + "\n")
                    file_result.write("Window Size : " + str(window_size_value) + " ")
                    file_result.write("Stride Value : " + str(stride_value) + " Ceps : N/A Filter 22")
                    file_result.write("Kernel : " + kernel + " Noise Percentage : " + str(0) + " ")
                    file_result.write("With Log and Squared Changes with 20 log\n\n")

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
                                        # if idx % impurity == 0:
                                        #     noise_index = ri(1, 6)
                                        #     noise_fs, noise_data = wavfile.read('_background_noise_/'+noise_files[noise_index][:len(data)])
                                        #     data += noise_data[:len(data)]
                                    else:
                                        print(arsh)

                                    # for i in range(1, len(data)):
                                    #     data[i] -= 0.97*data[i-1]
                                    data = np.append(data[0], data[1:] - 0.97*data[:-1])

                                    Xs = np.zeros((int((len(data) - window_size)/stride) + 1, int(window_size/2)+1))
                                    window = get_window("hann", window_size, fftbins=True)

                                    for i in range(Xs.shape[0]):
                                        Xs[i] = np.abs(np.fft.rfft(data[i*stride:i*stride+window_size] * window)[:int(window_size/2)+1])

                                    Xs = (1.0/window_size) * (Xs ** 2)

                                    freq_min = 0
                                    freq_max = fs / 2
                                    mel_filter_num = 22

                                    mels = np.linspace(freq_to_mel(freq_min), freq_to_mel(freq_max), mel_filter_num+2)
                                    freqs = mel_to_freq(mels)

                                    filter_points = np.floor((window_size + 1) / fs * freqs).astype(int)

                                    filters = np.zeros((len(filter_points) - 2, int(window_size / 2 + 1)))

                                    for n in range(len(filter_points) - 2):
                                        filters[n, filter_points[n]:filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] -
                                                                                                        filter_points[n])
                                        filters[n, filter_points[n + 1]:filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] -
                                                                                                            filter_points[n + 1])

                                    Xs_filtered = 20.0 * np.log10(np.dot(filters, np.transpose(Xs)) + eps)

                                    # num_ceps = 12
                                    mfcc = dct(Xs_filtered, type=2, axis=1, norm='ortho')

                                    train_X.append(mfcc)
                                    train_Y.append(mapping_output[directory])

                    train_X = np.array(train_X)
                    train_Y = np.array(train_Y)
                    print(train_X.shape, train_Y.shape)
                    # pickle.dump(train_X, open('Q2_train_X_Submission.pkl', 'wb'))
                    # pickle.dump(train_Y, open('Q2_train_Y_Submission.pkl', 'wb'))

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

                                    # for i in range(1, len(data)):
                                    #     data[i] -= 0.97 * data[i - 1]

                                    data = np.append(data[0], data[1:] - 0.97 * data[:-1])

                                    window_size = int(0.001 * 15 * fs)
                                    stride = int(0.001 * 15 * 75 * 0.01 * fs)
                                    Xs = np.zeros((int((len(data) - window_size) / stride) + 1, int(window_size / 2) + 1))
                                    window = get_window("hann", window_size, fftbins=True)

                                    for i in range(Xs.shape[0]):
                                        Xs[i] = np.abs(
                                            np.fft.rfft(data[i * stride:i * stride + window_size] * window)[:int(window_size / 2) + 1])

                                    Xs = (1.0 / window_size) * (Xs ** 2)

                                    freq_min = 0
                                    freq_max = fs / 2
                                    mel_filter_num = 22

                                    mels = np.linspace(freq_to_mel(freq_min), freq_to_mel(freq_max), mel_filter_num + 2)
                                    freqs = mel_to_freq(mels)

                                    filter_points = np.floor((window_size + 1) / fs * freqs).astype(int)

                                    filters = np.zeros((len(filter_points) - 2, int(window_size / 2 + 1)))

                                    for n in range(len(filter_points) - 2):
                                        filters[n, filter_points[n]:filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] -
                                                                                                        filter_points[n])
                                        filters[n, filter_points[n + 1]:filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] -
                                                                                                            filter_points[n + 1])

                                    Xs_filtered = 20.0 * np.log10(np.dot(filters, np.transpose(Xs)) + eps)

                                    # num_ceps = 12
                                    mfcc = dct(Xs_filtered, type=2, axis=1, norm='ortho')

                                    test_X.append(mfcc)
                                    test_Y.append(mapping_output[directory])

                    test_X = np.array(test_X)
                    test_Y = np.array(test_Y)
                    # pickle.dump(test_X, open('Q2_test_X_Submission.pkl', 'wb'))
                    # pickle.dump(test_Y, open('Q2_test_Y_Submission.pkl', 'wb'))
                    print(test_X.shape, test_Y.shape)

                    # train_X = pickle.load(open('Q2_train_X.pkl', 'rb'))
                    # train_Y = pickle.load(open('Q2_train_Y.pkl', 'rb'))
                    # test_X = pickle.load(open('Q2_test_X.pkl', 'rb'))
                    # test_Y = pickle.load(open('Q2_test_Y.pkl', 'rb'))

                    train_X = normalize(train_X.reshape(train_X.shape[0], -1))
                    test_X = normalize(test_X.reshape(test_X.shape[0], -1))

                    print(train_X.shape, train_Y.shape)
                    print(test_X.shape, test_Y.shape)

                    print("Start Training")
                    svm_model_linear = SVC(kernel='linear', C=1).fit(train_X, train_Y)
                    print("Done Training")
                    pickle.dump(svm_model_linear, open('Q2_model_'+str(version_count)+'.pkl', 'wb'))

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
