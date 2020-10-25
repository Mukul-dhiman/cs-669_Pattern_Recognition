
import numpy as np
import matplotlib.pyplot as pt
import math

#read class1 data
data_class1 = open("../real_world_data/class1.txt",'r')

#read class2 data
data_class2 = open("../real_world_data/class2.txt",'r')

#read class3 data
data_class3 = open("../real_world_data/class3.txt",'r')

# List of class1 and class2 data 
data_class1_list = []
data_class2_list = []
data_class3_list = []



line = data_class1.readline()
while line:
    line = line.strip().split(" ")
    line[0] = float(line[0])
    line[1] = float(line[1])
    data_class1_list.append(line)
    line=data_class1.readline()
    
data_class1.close()

line = data_class2.readline()

while line:
    line = line.strip().split(" ")
    line[0] = float(line[0])
    line[1] = float(line[1])
    data_class2_list.append(line)
    line=data_class2.readline()
    
data_class2.close()

line = data_class3.readline()

while line:
    line = line.strip().split(" ")
    line[0] = float(line[0])
    line[1] = float(line[1])
    data_class3_list.append(line)
    line=data_class3.readline()
    
data_class3.close()

data_class1_list = np.array(data_class1_list)
data_class2_list = np.array(data_class2_list)
data_class3_list = np.array(data_class3_list)




print("----------------Initial Data----------------------")
print("Red is Class 1, Green is Class 2, Blue is Class 3")
pt.scatter(data_class1_list[:,0],data_class1_list[:,1],c='red',label='class1')
pt.scatter(data_class2_list[:,0],data_class2_list[:,1],c='green',label='class2')
pt.scatter(data_class3_list[:,0],data_class3_list[:,1],c='blue',label='class3')
pt.show()



np.random.shuffle(data_class1_list)
np.random.shuffle(data_class2_list)
np.random.shuffle(data_class3_list)

data_class1_list_70_per = data_class1_list[:1500]
data_class2_list_70_per = data_class2_list[:1500]
data_class3_list_70_per = data_class3_list[:1500]



data_class1_list_30_per = data_class1_list[:1500]
data_class2_list_30_per = data_class2_list[:1500]
data_class3_list_30_per = data_class3_list[:1500]





print("--------------70% of Initial Data-------------------")
print("Red is Class 1, Green is Class2, Blue is Class 3")
pt.scatter(data_class1_list_70_per[:,0],data_class1_list_70_per[:,1],c='red',label='class1')
pt.scatter(data_class2_list_70_per[:,0],data_class2_list_70_per[:,1],c='green',label='class2')
pt.scatter(data_class3_list_70_per[:,0],data_class3_list_70_per[:,1],c='blue',label='class3')
pt.show()


class Model:
    data_class1=0
    data_class2=0
    data_class3=0
    class1_mu = 0
    class2_mu = 0
    class3_mu = 0
    covariance_class1=0
    covariance_class2=0
    covariance_class3=0
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
    
    def probability_of_class3(self,x,y):
        cofficient = (2*math.pi)*(abs(np.linalg.det(self.covariance_class3)))
        covariance_inverse = (self.covariance_class3)**(-1)
        x_base_mu = [x-self.class3_mu[0],y-self.class3_mu[1]]
        x_base_mu_t = np.atleast_2d(x_base_mu).T
        exp_pow = np.dot(x_base_mu, (np.dot(covariance_inverse, x_base_mu_t)))
        return cofficient+(-1)*(1/2)*(exp_pow[0])
    
    def predict(self,x,y):
        pro_in_class1 = self.probability_of_class1(x,y)
        pro_in_class2 = self.probability_of_class2(x,y)
        pro_in_class3 = self.probability_of_class3(x,y)
        if(pro_in_class1>=pro_in_class2 and pro_in_class1>=pro_in_class3):
            return 1
        elif(pro_in_class2>=pro_in_class3):
            return 2
        else:
            return 3
    
    def accuracy(self,data1,class1,data2,class2,data3,class3):
        correct=0
        total=len(data1)+len(data2)+len(data3)
        for i in range(len(data1)):
            if(self.predict(data1[i][0],data1[i][1])==class1):
                correct=correct+1
        for i in range(len(data2)):
            if(self.predict(data2[i][0],data2[i][1])==class2):
                correct=correct+1
        for i in range(len(data3)):
            if(self.predict(data3[i][0],data3[i][1])==class3):
                correct=correct+1
        return (correct/total)*100
            
    
    def __init__(self,data_class1,data_class2,data_class3):
        self.data_class1=data_class1
        self.data_class2=data_class2
        self.data_class3=data_class3
        self.class1_mu = data_class1.mean(axis=0)
        self.class2_mu = data_class2.mean(axis=0)
        self.class3_mu = data_class3.mean(axis=0)
        self.covariance_class1=np.cov(data_class1[0],data_class1[1])
        self.covariance_class2=np.cov(data_class2[0],data_class2[1])
        self.covariance_class3=np.cov(data_class3[0],data_class3[1])
    
m = Model(data_class1_list_70_per,data_class2_list_70_per,data_class3_list_70_per)

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

def scale3(x,y):
    z = []
    for i in range(len(x)):
            z.append(m.probability_of_class3(x[i],y[i]))
    return z

x_d = []
y_d = []
mn=0
for i in range(400):
    mn2=0
    for j in range(400):
        x_d.append(mn)
        y_d.append(mn2)
        mn2=mn2+5
    mn=mn+5

fig = pt.figure()

z = scale1(x_d,y_d)
ax = pt.axes(projection ="3d")
ax.scatter(x_d, y_d, z, c='r', marker='o')
pt.show()

z = scale2(x_d,y_d)
ax = pt.axes(projection ="3d")
ax.scatter(x_d, y_d, z, c='g', marker='^')
pt.show()

z = scale3(x_d,y_d)
ax = pt.axes(projection ="3d")
ax.scatter(x_d, y_d, z, c='b', marker='^')


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

print("accuracy:",m.accuracy(data_class1_list_30_per,1,data_class2_list_30_per,2,data_class3_list_30_per,3))