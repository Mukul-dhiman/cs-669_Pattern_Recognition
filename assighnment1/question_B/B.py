
import numpy as np
import matplotlib.pyplot as pt
import math

#read class1 data
data_class1 = open("../non_linearly_seperable_data/Class1.txt",'r')

#read class2 data
data_class2 = open("../non_linearly_seperable_data/Class2.txt",'r')


# List of class1 and class2 data 
data_class1_list = []
data_class2_list = []




line = data_class1.readline()
while line:
    line = line.strip().split("\t")
    line[0] = float(line[0])
    line[1] = float(line[1])
    data_class1_list.append(line)
    line=data_class1.readline()
    
data_class1.close()

line = data_class2.readline()

while line:
    line = line.strip().split("\t")
    line[0] = float(line[0])
    line[1] = float(line[1])
    data_class2_list.append(line)
    line=data_class2.readline()
    
data_class2.close()


data_class1_list = np.array(data_class1_list)
data_class2_list = np.array(data_class2_list)



print("----------------Initial Data----------------------")
print("Red is Class 1, Green is Class2")
pt.scatter(data_class1_list[:,0],data_class1_list[:,1],c='red',label='class1')
pt.scatter(data_class2_list[:,0],data_class2_list[:,1],c='green',label='class2')
pt.show()



np.random.shuffle(data_class1_list)
np.random.shuffle(data_class2_list)

data_class1_list_70_per = data_class1_list[:700]
data_class2_list_70_per = data_class2_list[:700]



data_class1_list_30_per = data_class1_list[:700]
data_class2_list_30_per = data_class2_list[:700]





print("--------------70% of Initial Data-------------------")
print("Red is Class 1, Green is Class2")
pt.scatter(data_class1_list_70_per[:,0],data_class1_list_70_per[:,1],c='red',label='class1')
pt.scatter(data_class2_list_70_per[:,0],data_class2_list_70_per[:,1],c='green',label='class2')
pt.show()


class Model:
    data_class1=0
    data_class2=0
    class1_mu = 0
    class2_mu = 0
    covariance_class1=0
    covariance_class2=0
    def probability_of_class1(self,x,y):
        cofficient = (2*math.pi)*(abs(np.linalg.det(self.covariance_class1)))
        covariance_inverse = (self.covariance_class1)**(-1)
        x_base_mu = [x-self.class1_mu[0],y-self.class1_mu[1]]
        x_base_mu_t = np.atleast_2d(x_base_mu).T
        exp_pow = np.dot(x_base_mu, (np.dot(covariance_inverse, x_base_mu_t)))
        return cofficient+(-1)*(1/2)*(exp_pow[0])
        
        
    def probability_of_class2(self,x,y):
        cofficient = (2*math.pi)*(abs(np.linalg.det(self.covariance_class2)))
        covariance_inverse = (self.covariance_class2)**(-1)
        x_base_mu = [x-self.class2_mu[0],y-self.class2_mu[1]]
        x_base_mu_t = np.atleast_2d(x_base_mu).T
        exp_pow = np.dot(x_base_mu, (np.dot(covariance_inverse, x_base_mu_t)))
        return cofficient+(-1)*(1/2)*(exp_pow[0])
    
    def predict(self,x,y):
        pro_in_class1 = self.probability_of_class1(x,y)
        pro_in_class2 = self.probability_of_class2(x,y)
        if(pro_in_class1>pro_in_class2):
            return 1
        else:
            return 2
    
    def accuracy(self,data1,class1,data2,class2):
        correct=0
        total=len(data1)+len(data2)
        for i in range(len(data1)):
            if(self.predict(data1[i][0],data1[i][1])==class1):
                correct=correct+1
        for i in range(len(data2)):
            if(self.predict(data2[i][0],data2[i][1])==class2):
                correct=correct+1
        return (correct/total)*100
            
    
    def __init__(self,data_class1,data_class2):
        self.data_class1=data_class1
        self.data_class2=data_class2
        self.class1_mu = data_class1.mean(axis=0)
        self.class2_mu = data_class2.mean(axis=0)
        self.covariance_class1=np.cov(data_class1[0],data_class1[1])
        self.covariance_class2=np.cov(data_class2[0],data_class2[1])
    
m = Model(data_class1_list_70_per,data_class2_list_70_per)

# For graph
def scale1(x,y):
    z = []
    for i in range(len(x)):
        z.append(m.probability_of_class1(x[i],y[i]))
    return z

def scale2(x,y):
    z = []
    for i in range(len(x)):
            z.append(m.probability_of_class2(x[i],y[i]))
    return z

x_d = []
y_d = []
mn=-100
for i in range(400):
    mn2=-100
    for j in range(400):
        x_d.append(mn)
        y_d.append(mn2)
        mn2=mn2+0.5
    mn=mn+0.5

fig = pt.figure()

z = scale1(x_d,y_d)
ax = pt.axes(projection ="3d")
ax.scatter(x_d, y_d, z, c='r', marker='o')
pt.show()

z = scale2(x_d,y_d)
ax = pt.axes(projection ="3d")
ax.scatter(x_d, y_d, z, c='g', marker='^')
pt.show()

while(1):
    print("enter E to exit or C to continue")
    chose=input()
    if(chose=='C'):
        print("enter value of x for prediction")
        x=int(input())
        print("enter value of y for prediction")
        y=int(input())
        print("point (x,y) belong to class",m.predict(x,y))
    else:
        break



print("accuracy:",m.accuracy(data_class1_list_30_per,1,data_class2_list_30_per,2))


