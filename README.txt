Submitted by:
Nathan John Kibanoff
Don Richson Que
Jasper Domenique Tadeo
Edvi Jay Yap

Raw features used: NumPedestrianVictim, NumDeath, NumInjured, NumVehInteraction, IsInJunction
Among all the features shown in the training dataset, we chose those five because they seemed like the most probable factors that are important in determining the severity of vehicle accidents

How to run
	Java implementation
		1. Compile Neurons.java, Weights.java, NeuralNetwork.java, and Program.java by opening command prompt on the folder where the files are located and type "javac *.java" (without the quotation marks)
		2. Run Program.java by typing "java Program" (without the quotation marks) on the command line
		3. While the program is running, it will display the necessary instructions, and the user must provide the input in proper format
			- When manually inputting weights between layers n and n+1 with sizes topologies t(n) and t(n+1) respectively, it is recommended to input them in a t(n) x t(n+1) matrix format, with a single space separating two distinct values, and an endline ('\n') after each row.
		4. For each datapoint per epoch, the following data will be printed out in the following order:
			FEED FORWARD
			- Values of neurons in first hidden layer
			- Weights between input layer and first hidden layer
			For n>1
				- Values of neurons in nth layer
				- Weights between (n-1)th hidden layer and nth layer
			- Error values for current datapoint
			- Total error for current datapoint
			BACKWARD PROPAGATION
			- Derivatives of output layer after feed forward
			- Gradients of output layer and preceding layers (excluding input)
		5. After each epoch, the program will print out the overall error rate (average error rates of all data points in the epoch)
	Python implementation
		*NOTE: The following were used in this implementation. We're unsure if it can run on earlier versions:
			- Python 3.6.3
			- Keras 2.1.1
			- matplotlib 2.1.0
			- numpy 1.13.3
			- Theano 1.0.0
			*By default, Keras uses tensorflow backend. Go to the location where keras.json is located (usually in C:\Users\<username>\.keras) and change the "backend" value to "theano"
		1. Go to the command prompt and type "python NeuralNetwork.py" (without quotation marks)
		2. The program will output the total loss value and accuracy for each datapoint
		3. Additionally, it will also output three graphs: Error rate per epoch during training, Experiment A, and Experiment B

Model configuration
	There are comments found in NeuralNetwork.py and NeuralNetwork.java which describe how the functions work, and how the model can be configured