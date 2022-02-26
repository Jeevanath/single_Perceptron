#Perceptron

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1,10, num=10)

y = []
w= []



# y = [5,9,10,8,2,4,5,9,3,12]

for i in range(0,len(x)):
	w.append(3*x[i] + 2)

y = [10, 15 ,2, 9, 20, 0, 31, 12, 5, 22]

print(x)
print(y)

z = [1,1,0,0,1,0,1,0,0,0]

# plt.plot(x,w)
# plt.scatter(x,y)
# plt.show()




class Perceptron(object):
	def __init__(self, no_of_inputs, threshold=1000, learning_rate=0.001):
		self.threshold = threshold
		self.learning_rate = learning_rate
		self.weights = np.zeros(no_of_inputs + 1)
	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		B = np.where(summation  > 0,1,0)
		return B

	def train(self, training_inputs, labels):
		for _ in range(self.threshold):
			for inputs, label in zip(training_inputs, labels):
				prediction = self.predict(inputs)
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs
				self.weights[0] += self.learning_rate * (label - prediction)


A = Perceptron(no_of_inputs=2)




# x = np.array(x,y)

d=np.vstack((x,y)).T
print(d)

A.train(training_inputs = d, labels = z )






coords = np.random.rand(100, 2) * 30

def xysplit(ary):
    return ary[:,:-1], ary[:,-1]

X, Y = xysplit(coords)

ans = A.predict(coords)


print(ans)



fig, ax = plt.subplots()

colours = ('lightgreen', 'lightblue')
for i in range(100):
    ax.scatter(X[i], 
               Y[i], 
               c=colours[ans[i]])
plt.plot()
plt.plot(x,w)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Single Layer Perceptron for Linear classification")
plt.show()


