
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