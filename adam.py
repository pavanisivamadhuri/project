class SigmoidNeuronAdam:
  
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
  
  #def fit(self, X, Y, epochs=100, eta=0.01, gamma=1.9, mini_batch_size=100, eps=1e-5,  
   #       beta=0.9, beta1=0.1, beta2=0.199 ):
  def fit(self, X, Y, epochs=10, eta=0.0014,  eps=1e-6, beta1=0.7, beta2=0.999 ): 
    # initialise w, b
    #if initialise:
    self.w = np.random.randn(1, X.shape[1])
    self.b = 0
      
    
    loss = []
    
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
      
      
      v_w, v_b = 0, 0
      m_w, m_b = 0, 0
      num_updates = 0


      dw = 0
      db = 0
      for x, y in zip(X, Y):
          dw = self.grad_w(x, y)
          db = self.grad_b(x, y)
          num_updates += 1
          m_w = beta1 * m_w + (1-beta1) * dw
          m_b = beta1 * m_b + (1-beta1) * db
          v_w = beta2 * v_w + (1-beta2) * dw**2
          v_b = beta2 * v_b + (1-beta2) * db**2
          m_w_c = m_w / (1 - np.power(beta1, num_updates))
          m_b_c = m_b / (1 - np.power(beta1, num_updates))
          v_w_c = v_w / (1 - np.power(beta2, num_updates))
          v_b_c = v_b / (1 - np.power(beta2, num_updates))
          self.w -= (eta / np.sqrt(v_w_c) + eps) * m_w_c
          self.b -= (eta / np.sqrt(v_b_c) + eps) * m_b_c
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
    sna = SigmoidNeuronAdam()
l=sna.fit(X_train, y_train)
#print(len(l))
a=np.asarray(l)
#print(a.shape)
plt.plot(l)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

mi=min(l)
for i in range(len(l)):
  if mi==l[i]:
    #print(i)
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
  #print(y_pred_test)
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
import math
import sklearn.metrics
# saving model weights 

wa=sna.w
ba=sna.b
wa=wa.ravel()
print(wa,ba)

wg=[-1.37326115,  3.06486133, -4.56618867, -4.12885534, -2.82597038 , 4.15793448 ,-0.61586065 ,-0.08696289, -0.8864412,  -0.40469274] 
bg = [1.43206064]
wa=[-1.17336511  ,0.57982791, -0.96265004  ,0.34608458 ,-1.71589575  ,0.5666959 ,-1.23674197, -0.76984886,  0.20949419,  0.05584285] 
ba=[0.00374794]
