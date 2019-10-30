import random
import math as m
import numpy as np
import operator
import matplotlib.pyplot as plt
import sys


#read Input dataset and shuffle the rows
def readdatafile(filename):
	with open(filename, 'r') as ifl:
		array = [l.strip().split(',') for l in ifl]
		dataset = list(array)[:-1]
		random.shuffle(dataset)

		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
	return dataset


def euclidean_Distance(x1, x2, length):
	dist = 0
	for i in range(length):
		dist += np.square(x1[i] - x2[i])
	return np.sqrt(dist)


def cosine_similarity(x1,x2,length):
	numerator = 0
	x =0
	y = 0
	for i in range(length):
		numerator += x1[i]*x2[i]
		x += np.square(x1[i])
		y += np.square(x2[i])
	denominator = (np.sqrt(x))*(np.sqrt(y))
	similarity = numerator / denominator
	return similarity


def avg_cm(x_list):
    sum = 0
    for x in x_list:
        sum = sum + x
    return sum/len(x_list)


#finds the nearest k data sets for each test instances
def findneighbors(train_data, test_data, k,metric):
	distances = []
	neighbors = []
	length = len(test_data)-1
	for i in range(len(train_data)):
		if metric == "Euclidean":
			distance = euclidean_Distance(test_data, train_data[i], length)
		else:
			distance = cosine_similarity(test_data, train_data[i], length)

		distances.append((train_data[i], distance))
	if metric == "Euclidean":
		#sorting the dataset based on the distance to find neighbors
		distances.sort(key=operator.itemgetter(1))
	else:
		distances.sort(key=operator.itemgetter(1),reverse= True)
	#making list of k smallest distance neighboring dataset
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


#finding the number of neighbour of each classes for the dataset
#the class with maximum number of dataset is the prediction class
def classification(neighbors):
	count = {}
	for x in range(len(neighbors)):
		prediction_type = neighbors[x][-1]
		if prediction_type in count:
			count[prediction_type] += 1
		else:
			count[prediction_type] = 1
	predictionclass = max(count.items(), key = operator.itemgetter(1))[0]
	return predictionclass


def calc_Accuracy(test_data, predictions):
	correct = 0
	#last element of test set is real classification,
	#so making list of true classification
	actual = [instances[-1] for instances in test_data]
	#comparing each values in list if they are similar
	for x, y in zip(actual, predictions):
		if x == y:
			correct += 1
	accuracy = (correct / float(len(actual))) * 100
	return accuracy


def find_confusion_matrix(predictions, actual):
	actual_class = []
	# last element of test set is real classification,
	# so making list of true classification
	actual_class = [instances[-1] for instances in actual]
	num_of_classes = len(set(actual_class))
	# replacing the class name with integer value
	p = list(map(lambda x: 0 if x == "Iris-setosa" else x, predictions))
	p = list(map(lambda x: 1 if x == "Iris-versicolor" else x, p))
	p = list(map(lambda x: 2 if x == "Iris-virginica" else x, p))

	a = list(map(lambda x: 0 if x == "Iris-setosa" else x, actual_class))
	a = list(map(lambda x: 1 if x == "Iris-versicolor" else x, a))
	a = list(map(lambda x: 2 if x == "Iris-virginica" else x, a))

	#print(p,a)
	confusion_matrix = np.zeros((num_of_classes,num_of_classes))
	for i in range(len(actual_class)):
		confusion_matrix[p[i]][a[i]] += 1
	print('confusion matrix:',confusion_matrix)
	return confusion_matrix


#splits the dataset in test and training set as per 5 fold cross validataion and does classification
def knn_CrossValidation(dataset_shuffled,k,cv,metric):
	total_sample = len(dataset_shuffled)
	num_TestSample = int((1 / cv) * total_sample)
	t1 = 0
	t2 = num_TestSample
	accuracies =[]
	confusion_matrices =[]
	for i in range(cv):
		predictions = []
		#spliting the dataset in test and train set
		test_data = dataset_shuffled[t1:t2]
		train_data = dataset_shuffled[0:t1] + dataset_shuffled[t2:total_sample]
		t1 = t1 + num_TestSample
		t2 = t2 + num_TestSample

		#for each testset performing classification
		for x in range(len(test_data)):
			neighbors = findneighbors(train_data, test_data[x], k, metric)
			prediction = classification(neighbors)
			predictions.append(prediction)

		confusion_matrix = find_confusion_matrix(predictions, test_data)
		confusion_matrices.append(confusion_matrix)
		accuracy = calc_Accuracy(test_data,predictions)
		accuracies.append(accuracy)

	average_confusion_matrix = avg_cm(confusion_matrices)
	print('Accuracies:',accuracies)
	average_accuracy = sum(accuracies) / len(accuracies)
	print('Average accuracy:',average_accuracy)
	print('Best confusion matrix:',average_confusion_matrix)
	plotConfusionMatrix(test_data, predictions, average_confusion_matrix, normalize=True, title=None, cmap=None, plot=True)

def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):
    y_true = []
	#create a list of target in test dataset
    for x in range(len(test_set)):
        test = test_set[x][-1]
        y_true.append(test)

	# Find out the unique classes
    classes = list(np.unique(list(y_true)))

    if cmap is None:
        cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted ')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()


if __name__== "__main__":
   dataset_shuffled = readdatafile('iris.data')
   k = int(sys.argv[2])
   cv = 5
   metric = sys.argv[-1]
   knn_CrossValidation(dataset_shuffled,k,cv,metric)





