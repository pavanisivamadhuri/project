from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print(model.predict(X_test))
print(y_test)
%%time
model.score(X_train,y_train)
model.score(X_test,y_test)
import sklearn.metrics
class SigmoidNeuron:
  
  def __init__(self):
    self.w = None
    self.b = None
    
  def perceptron(self, x):
    return np.dot(x, self.w.T) + self.b
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def grad_w(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred) * x
  
  def grad_b(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred)
  
  def fit(self, X, Y, epochs=1000, learning_rate=1, initialise=True):
    
    # initialise w, b
    if initialise:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0
      
    
    loss = []
    
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
      dw = 0
      db = 0
      for x, y in zip(X, Y):
        dw += self.grad_w(x, y)
        db += self.grad_b(x, y)       
      self.w -= learning_rate * dw
      self.b -= learning_rate * db
      Y_pred = self.sigmoid(self.perceptron(X))
      Y_pred=Y_pred.ravel()
      ce=sklearn.metrics.log_loss(y_train, Y_pred)
      loss.append(ce)

    return loss
  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.sigmoid(self.perceptron(x))
      Y_pred.append(y_pred)
    return np.array(Y_pred)
  
  sn = SigmoidNeuron()
l=sn.fit(X_train, y_train, 500, 0.05, True)
print(len(l))
a=np.asarray(l)
print(a.shape)
plt.plot(l)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

mi=min(l)
for i in range(len(l)):
  if mi==l[i]:
    print(i)
    break

y_pred_train=sn.predict(X_train)
#print (y_pred_train)
for i in range(len((y_pred_train)-1)):
  if y_pred_train[i]>0.5:
    y_pred_train[i]=1
  else:
    y_pred_train[i]=0
y_pred_train=y_pred_train.ravel()
#print(y_pred_train)

for each in y_train:
  if each>0.5:
    each=1
  else:
    each=0
y_train=y_train.ravel()

#print((y_pred_train),( y_train))  


count=0
total=len(y_train)-1
for i in range(total):
  if  y_train[i]==y_pred_train[i]:
    count=count+1

print("Train accuracy is : ", count/total)

y_pred_test=sn.predict(X_test)

for i in range(len(y_pred_test)):
  if y_pred_test[i]>0.5:
    y_pred_test[i]=1
  else:
    y_pred_test[i]=0 

# print(y_pred_test)

# for test accuracy
count=0
total=len(y_test)-1
for i in range(total):
  if  y_test[i]==y_pred_test[i]:
    count=count+1

print("Test accuracy is : ", count/total)
print(sn.w,sn.b)
# saving model weights for epoch=2000, learning rate=0.85

wg=sn.w
bg=sn.b
wg=wg.ravel()
print(wg,bg)
wg=[-1.37326115,  3.06486133, -4.56618867, -4.12885534, -2.82597038 , 4.15793448 ,-0.61586065 ,-0.08696289, -0.8864412,  -0.40469274] 
bg = [1.43206064]
