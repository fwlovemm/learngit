#ID3决策算法简单应用
#2018.10.20
from sklearn.feature_extraction import DictVectorizer
import csv

from sklearn import preprocessing
from sklearn import tree
#读取和处理数据
alldata = open(r'AllElectronics.csv','r')
reader = csv.reader(alldata)
reader = list(reader)
headers = reader[0]
reader = reader[1:]

#构建特征值和标签
featureList = []
labelList = []
for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1,5):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)
print(labelList)

#特征值转化0 1
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()
print(dummyX)

#标签转化0 1
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)

#构建ID3分类器进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print(clf)

#构建新数据
oneRowX = dummyX[0,:]
oneRowX[1] = 1
oneRowX[2] = 0

#进行预测
predictedY = clf.predict(oneRowX.reshape(1,-1))
print("predictedY: " + str(predictedY))







