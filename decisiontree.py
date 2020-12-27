from sklearn.tree import DecisionTreeClassifier
import preprocess
import sys
file = open('./figures/decision_tree_output.txt', 'w')
sys.stdout = file

if __name__ == '__main__':
    classifier = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    preprocessor = preprocess.Preprocessor()
    ratio = 0.7
    # preprocessor.visualize_data()
    preprocessor.tokenize_data()
    training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]

    classifier.fit(training_data_x, training_data_y)
    predict = classifier.predict(data_x)
    wrong = 0
    for i in range(len(predict)):
        if predict[i] != data_y[i]:
            wrong += 1
    print('using summary, accuracy : ', (1 - wrong / len(predict)))

    training_data_x = preprocessor.sequenced_titles[:int(ratio * len(preprocessor.sequenced_titles))]
    data_x = preprocessor.sequenced_titles[int(ratio * len(preprocessor.sequenced_titles)):]

    classifier.fit(training_data_x, training_data_y)
    predict = classifier.predict(data_x)
    wrong = 0
    for i in range(len(predict)):
        if predict[i] != data_y[i]:
            wrong += 1
    print('using title, accuracy : ', (1 - wrong / len(predict)))
    file.close()
