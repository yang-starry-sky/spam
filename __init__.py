import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import joblib


def load_train_data(path="./train/"):
    train_set = []
    train_label = []
    for i in range(2):
        for filename in list(os.listdir(path + str(i))):
            with open(path + str(i) + '/' + filename, "r") as f:
                train_set.append(f.read())
            train_label.append(i)
    train_set = np.array(train_set)
    train_label = np.array(train_label)
    # 打乱
    index = np.arange(len(train_label))
    np.random.shuffle(index)
    train_set = train_set[index]
    train_label = train_label[index]
    return train_set, train_label


def train():
    train_set, train_label = load_train_data()
    vectorizer = TfidfVectorizer()
    train_set_vec = vectorizer.fit_transform(train_set)
    with open('./TfidfVectorizer.pkl', 'wb') as f:
        joblib.dump(vectorizer, f)
    train_set_vec, test_set_vec, train_label, test_label = train_test_split(train_set_vec, train_label, test_size=0.2)
    clf = MultinomialNB().fit(train_set_vec, train_label)
    joblib.dump(clf, "./spam_classification.pkl")
    output_performance(test_label, test_set_vec, clf)


def load_test_data(path="./test/"):
    test_set = []
    file_name = os.listdir(path)
    for filename in file_name:
        with open(path + filename, "r") as f:
            test_set.append(f.read())
    test_set = np.array(test_set)
    return test_set, file_name


def output_performance(test_label, test_set_vec, clf):
    predict = clf.predict(test_set_vec)
    print('Accuracy score: ', format(accuracy_score(test_label, predict)))
    print('Precision score: ', format(precision_score(test_label, predict)))
    print('Recall score: ', format(recall_score(test_label, predict)))
    print('F1 score: ', format(f1_score(test_label, predict)))
    print(classification_report(test_label, predict))


def output_result(clf, result_file='./result.txt'):
    test_set, fileName = load_test_data()
    vectorizer=joblib.load('./TfidfVectorizer.pkl')
    test_set_vec = vectorizer.transform(test_set)
    predict = clf.predict(test_set_vec)
    with open(result_file, "w") as f:
        for i in range(len(predict)):
            f.write(str(predict[i]) + '/' + fileName[i] + '\n')


def main():
    train()
    if os.path.exists('spam_classification.pkl'):
        clf = joblib.load('./spam_classification.pkl')
        output_result(clf)
    else:
        train()

# 运行
if __name__ == '__main__':
    main()
