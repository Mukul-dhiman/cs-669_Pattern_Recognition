
import numpy as np
import matplotlib.pyplot as pt

#read class1 data
data_class1 = open("../linearly_seperable_data/Class1.txt",'r')

#read class2 data
data_class2 = open("../linearly_seperable_data/Class2.txt",'r')


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
pt.scatter(data_class1_list[:,0],data_class1_list[:,1],c='red',label='class1')
pt.scatter(data_class2_list[:,0],data_class2_list[:,1],c='green',label='class2')
pt.show()
#print(data_class1_list)



np.random.shuffle(data_class1_list)
np.random.shuffle(data_class2_list)

data_class1_list_70_per = data_class1_list[:700]
data_class2_list_70_per = data_class2_list[:700]



data_class1_list_30_per = data_class1_list[700:]
data_class2_list_30_per = data_class2_list[700:]





print("--------------70% of Initial Data-------------------")
pt.scatter(data_class1_list_70_per[:,0],data_class1_list_70_per[:,1],c='red',label='class1')
pt.scatter(data_class2_list_70_per[:,0],data_class2_list_70_per[:,1],c='green',label='class2')
pt.show()









