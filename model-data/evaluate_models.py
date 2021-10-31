# Disclaimer: code in this file is based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

# Path to csv file
csv_path = "landmarks_classifier.csv"


def main():
    # Get training data; split training data into x and y training sets
    dataset = read_csv(csv_path, header=None)
    hand_shape_data = dataset.values
    x = hand_shape_data[:, 1:43]
    y = hand_shape_data[:, 0]

    # Split test data into training and validation sets
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    # Build and evaluate models
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]

    # Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare the different models using pyplot
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()


if __name__ == "__main__":
    main()
