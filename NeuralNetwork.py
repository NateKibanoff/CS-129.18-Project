#features used (original)
# NumPedestrianVictim, NumDeath, NumInjured, NumVehInteraction, IsInJunction
# After running the three autoencoders, NumDeath showed the highest error, therefore it is eliminated in Experiment B

from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(7)
dataset=numpy.loadtxt("Train.csv",delimiter=",")
auto1=numpy.loadtxt("Autoencoder1.csv",delimiter=",")
auto2=numpy.loadtxt("Autoencoder2.csv",delimiter=",")
auto3=numpy.loadtxt("Autoencoder3.csv",delimiter=",")
reduced=numpy.loadtxt("Test - reduced.csv",delimiter=",")

x_train=dataset[:70,0:5]
y_train=dataset[:70,5]
x_test=dataset[70:,0:5]
y_test=dataset[70:,5]

x_auto1=auto1[:,:5]
y_auto1=auto1[:,5:]
x_auto2=auto2[:,:5]
y_auto2=auto2[:,5:]
x_auto3=auto3[:,:5]
y_auto3=auto3[:,5:]

x_reduced_train=reduced[:70,0:4]
y_reduced_train=reduced[:70,4]
x_reduced_test=reduced[70:,0:4]
y_reduced_test=reduced[70:,4]

model=Sequential()
model.add(Dense(3,input_dim=5,activation="sigmoid"))
model.add(Dense(2,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
epochs=[]
errors=[]
for i in range(100):
	epochs.append(i+1)
	model.fit(x_train,y_train,epochs=1,batch_size=10)
	scores=model.evaluate(x_train,y_train)
	errors.append(scores[0])

g1=plt.figure(1)
plt.title("Error rate per epoch (training)")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.plot(epochs,errors)
plt.yticks([0.2,0.4,0.6,0.8,1.0])
plt.grid()
g1.show()

acc1=[]
for i in range(100):
	model.fit(x_test,y_test,epochs=1,batch_size=10)
	scores=model.evaluate(x_test,y_test)
	acc1.append(100*scores[1])

g2=plt.figure(2)
plt.title("Experiment A")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (percentage)")
plt.plot(epochs,acc1)
plt.yticks([20,40,60,80,100])
plt.grid()
g2.show()

#Autoencoder for y=1
reduce1=Sequential()
reduce1.add(Dense(4,input_dim=5,activation="relu"))
reduce1.add(Dense(5,activation="sigmoid"))
reduce1.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
reduce1.fit(x_auto1,y_auto1,epochs=100,batch_size=10)

#Autoencoder for y=2
reduce2=Sequential()
reduce2.add(Dense(4,input_dim=5,activation="relu"))
reduce2.add(Dense(5,activation="sigmoid"))
reduce2.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
reduce2.fit(x_auto2,y_auto2,epochs=100,batch_size=10)

#Autoencoder for y=3
reduce3=Sequential()
reduce3.add(Dense(4,input_dim=5,activation="relu"))
reduce3.add(Dense(5,activation="sigmoid"))
reduce3.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
reduce3.fit(x_auto3,y_auto3,epochs=100,batch_size=10)

#Classifier with reduced features
reduced_nn=Sequential()
reduced_nn.add(Dense(3,input_dim=4,activation="sigmoid"))
reduced_nn.add(Dense(2,activation="relu"))
reduced_nn.add(Dense(1,activation="sigmoid"))
reduced_nn.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
reduced_nn.fit(x_reduced_train,y_reduced_train,epochs=100,batch_size=10)
acc2=[]
for i in range(100):
	reduced_nn.fit(x_reduced_test,y_reduced_test,epochs=1,batch_size=10)
	scores=reduced_nn.evaluate(x_reduced_test,y_reduced_test)
	acc2.append(100*scores[1])

g3=plt.figure(3)
plt.title("Experiment B")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (percentage)")
plt.plot(epochs,acc2)
plt.yticks([20,40,60,80,100])
plt.grid()
plt.show()