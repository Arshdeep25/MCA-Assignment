import operator
import os
import pickle
import time
import numpy as np

for fileName in os.listdir('train/query'):
    file = open('train/query/'+fileName, 'r')
    queryFileName = file.read()
    queryFileName = queryFileName[queryFileName.find('_')+1:queryFileName.find(' ')]
    print('pickle/'+queryFileName+'.pkl')
    score = dict()
    queryFeatures = pickle.load(open('pickle/'+queryFileName+'.pkl', 'rb'))
    start_time = time.time()
    for savedFeaturesFileName in os.listdir('pickle'):
        start_time_loop = time.time()
        if savedFeaturesFileName.find(queryFileName) == -1:

            savedFeatures = pickle.load(open('pickle/'+savedFeaturesFileName, 'rb'))

            denominator = 1
            numerator = 0

            map_color_freq_1 = queryFeatures
            map_color_freq_2 = savedFeatures

            map_color_freq = dict()
            for key in map_color_freq_1:
                map_color_freq[key] = [map_color_freq_1[key]]

            for key in map_color_freq_2:
                if key in map_color_freq:
                    map_color_freq[key].append(map_color_freq_2[key])
                else:
                    map_color_freq[key] = [map_color_freq_2[key]]

            distance_score = 0

            for key in map_color_freq:
                if np.array(map_color_freq[key]).shape[0] > 1:
                    for k in range(4):
                        distance_score += abs(map_color_freq[key][0][k] - map_color_freq[key][1][k]) / (
                                    1 + map_color_freq[key][0][k] + map_color_freq[key][1][k])
                else:
                    for k in range(4):
                        distance_score += (map_color_freq[key][0][k]) / (1 + map_color_freq[key][0][k])

            unique_colors = len(list(map_color_freq.keys()))

            score[savedFeaturesFileName[:savedFeaturesFileName.find('.pkl')]] = distance_score/unique_colors

    numberResults = 0
    queryFileName = queryFileName[:queryFileName.find('.')]
    goodResult = open('train/ground_truth/'+fileName[:-10]+'_good.txt', 'r')
    goodResultData = goodResult.read().split("\n")[:-1]
    okResult = open('train/ground_truth/'+fileName[:-10]+'_ok.txt', 'r')
    okResultData = okResult.read().split("\n")[:-1]
    junkResult = open('train/ground_truth/'+fileName[:-10]+'_junk.txt', 'r')
    junkResultData = junkResult.read().split("\n")[:-1]
    numberResults += len(goodResultData) + len(okResultData) + len(junkResultData)

    # score = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
    sorted_score = dict(sorted(score.items(), key=operator.itemgetter(1), reverse=True))

    precision_good = 0
    recall_good = 0
    precision_ok = 0
    recall_ok = 0
    precision_junk = 0
    recall_junk = 0

    count_good = 0
    count_ok = 0
    count_junk = 0

    count = 0

    for key in sorted_score:
        count += 1
        if key in goodResultData:
            count_good += 1
        if key in okResultData:
            count_ok += 1
        if key in junkResultData:
            count_junk += 1
        if count >= numberResults:
            break

    precision_good = count_good/numberResults
    recall_good = count_good/len(goodResultData)

    precision_ok = count_ok / numberResults
    recall_ok = count_ok / len(okResultData)

    precision_junk = count_junk / numberResults
    recall_junk = count_junk / len(junkResultData)

    print(precision_good, recall_good)
    print(precision_ok, recall_ok)
    print(precision_junk, recall_junk)