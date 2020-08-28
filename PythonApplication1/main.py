from Net import Net
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

def main():
	

	training_iterations = 500
	training_set_count = 40
	new_net = 1

	print("Wczytywanie danych...")
	dataset = pd.read_csv("data/diagnosis.csv", sep=";", decimal=",")
	print("Przygotowanie danych...")
	X = dataset.iloc[:, 0:-2].values
	Y = dataset.iloc[:, -2:].values
	for n in range(len(X[0])):
		X[:,n] = LabelEncoder().fit_transform(X[:,n])
	scaler_X = StandardScaler()
	X = scaler_X.fit_transform(X)
	for n in range(len(Y[0])):
		Y[:,n] = LabelEncoder().fit_transform(Y[:,n])


	topology = [len(X[0]), 4, 4, len(Y[0])]
	
	print("Wczytywanie sieci...")
	try:
		filehandler = open("net.obj", 'rb') 
		myNet = pickle.load(filehandler)
	except:
		myNet = Net(topology)

	if new_net:
		myNet = Net(topology)
	
	print("Uczenie...")
	training_data_counter = list(range(training_set_count))
	for n in range(training_iterations):
		print("itteration: {} / {}".format(n+1, training_iterations))
		for data_index in training_data_counter:
			myNet.feed_forward(X[data_index])
			myNet.back_prop(Y[data_index])

		myNet.update_weights()
		pass



	correct_recognize_counter = 0
	results = []
	print("Sprawdzanie wynikow... ")
	training_data_counter = list(range(training_set_count, len(X)))
	for data_index in range(len(training_data_counter)):
		myNet.feed_forward(X[data_index])
		result = myNet.get_result()
		
		results.append(result)
		result = np.around(result, 0)
		#print(result)
		comparison = result == Y[data_index]
		if comparison.all():
			correct_recognize_counter+=1


	errors = myNet.getErrors()
	print(errors[0])
	print(errors[-1])

	plt.subplot(2, 1, 1)
	plt.xscale(value="linear")
	plt.plot(range(len(errors)), errors)
	plt.title("Global error", fontsize=12)
	plt.xlabel("Trained datasets count")
	plt.ylabel("Avarage error")

	plt.subplot(2, 1, 2)
	current_errors = errors[-training_iterations:]
	plt.plot(range(len(errors)-training_iterations,len(errors)), current_errors)
	plt.xlabel("Trained datasets count")
	plt.ylabel("Avarage error")
	plt.title("Current session error", fontsize=12)

	plt.suptitle("Efficiency: {} / {}, {}%".format(correct_recognize_counter, 120 - training_set_count, int(np.around(correct_recognize_counter/(120 - training_set_count), 2)*100)), fontsize=14)
	plt.tight_layout(pad=3) 


	filehandler = open("net.obj", 'wb') 
	pickle.dump(myNet, filehandler)


	plt.show()


	pass
	
main()