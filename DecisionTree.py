import collections
import numpy
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class DecisionNode:
    def __init__(self,attribiuteIndex=-1,targetClass=-1,NumericalSplit = -1):
        self.NumericalSplit = NumericalSplit
        self.attribiuteIndex=attribiuteIndex
        self.targetClass = targetClass
        self.children = {}

def Encode(X,Y,numericLoc):

    encodedFeature,X_encoders = [],[]
    le_Y = preprocessing.LabelEncoder()
    tX = numpy.transpose(X)

    for i in range(len(tX)):
        # if not numericLoc.__contains__(i):
        le_X = preprocessing.LabelEncoder()
        le_X.fit(tX[i])
        # print(list(le_X.classes_))
        transformed = le_X.transform(tX[i])
        if not numericLoc.__contains__(i):
            encodedFeature.append(transformed)
        else:
            encodedFeature.append(tX[i])
        X_encoders.append(le_X)

    le_Y.fit(Y)
    encodedLabel = le_Y.transform(Y)
    Y_encoders = le_Y

    return numpy.transpose(numpy.array(encodedFeature)),encodedLabel,X_encoders,Y_encoders

def Gain(X,Y,X_encoders=None,entropy=-1,attributeIndexList=None):

    entropyList = []

    for column in range(numpy.shape(X)[1]):
        en = entropy
        splitted = [[] for u in range(len(X_encoders[attributeIndexList[column]].classes_))] if not entropy==-1 else [[],[]]
        for entry in numpy.hstack((X,numpy.transpose(numpy.array([Y])))):
            splitted[int(entry[column])].append(entry[-1])

        for s in splitted:en-= len(s)/len(Y) * Entropy(s)
        entropyList.append(en)

    return entropyList.index(max(entropyList))

def Entropy(Y):
    return sum([-n * numpy.math.log2(n) for n in numpy.array(list(collections.Counter(Y).values())) / len(Y)])

def DecisionTree(X, Y, X_encoders, Y_encoders, attributeIndexList, threshold,numericLoc):

    if(len(attributeIndexList)==0 or len(X)<threshold): return DecisionNode(targetClass=max(set(Y), key=list(Y).count))

    entropy = Entropy(Y)

    if(entropy==0): return DecisionNode(targetClass=Y[0])

    X_copy = numpy.transpose(numpy.copy(X))

    for n in attributeIndexList:
        if attributeIndexList.__contains__(n):
            X_copy[attributeIndexList.index(n)]=BestSplit(X[:,attributeIndexList.index(n)],Y)
    X_copy=(numpy.transpose(X_copy))

    maxGainIndex = Gain(X_copy,Y,X_encoders,entropy,attributeIndexList)
    actualIndex = attributeIndexList[maxGainIndex]
    attributeIndexList.remove(actualIndex)

    splitted = [[] for u in range(len(X_encoders[actualIndex].classes_))] if not numericLoc.__contains__(actualIndex) else [[],[]]

    for entry in numpy.hstack((X_copy,numpy.transpose(numpy.array([Y])))):
        splitted[int(entry[maxGainIndex])].append(numpy.array(list(entry[:maxGainIndex])+list(entry[maxGainIndex+1:])))

    if not numericLoc.__contains__(actualIndex):
        decisionNode = DecisionNode(attribiuteIndex=actualIndex)
    else:
        print()
        decisionNode = DecisionNode(attribiuteIndex=actualIndex,NumericalSplit=100)
    # if numericLoc.__contains__(actualIndex): decisionNode.NumericalSplit = 100

    for i in range(len(splitted)):
        s = numpy.array(splitted[i])
        decisionNode.children[i] = DecisionNode(targetClass=max(set(Y), key=list(Y).count)) if len(s)==0 else DecisionTree(s[:,:-1],s[:,-1], X_encoders, Y_encoders, attributeIndexList.copy(), threshold,numericLoc)

    return decisionNode

def Traverse(rootNode):
    current = rootNode
    print(current.attribiuteIndex,current.targetClass)
    for key,value in current.children.items():
        Traverse(value)

def Predict(rootNode,encodedSample):
    if rootNode.attribiuteIndex == -1 or not rootNode.children.__contains__(encodedSample[rootNode.attribiuteIndex]):
        return rootNode.targetClass
    else:
        if not rootNode.NumericalSplit == -1:
            print(rootNode.NumericalSplit)
        return Predict(rootNode.children[encodedSample[rootNode.attribiuteIndex]],encodedSample) if rootNode.NumericalSplit == None else Predict(rootNode.children[1],encodedSample)


def Accuracy(rootNode,X_test,y_test):

    y_predict = numpy.array([Predict(rootNode,xt) for xt in X_test])
    return  accuracy_score(y_test, y_predict)*100

def BestSplit(numericData,Label):
    numericDataWithLabel = numpy.transpose(numpy.array([list(map(float, numericData)),Label]))
    allPossibles = []
    uniqeList = list(set(numericDataWithLabel[numericDataWithLabel[:, 0].argsort()][:,0]))
    for unique in uniqeList:
        numericDataEncoded = []

        for elem in numericDataWithLabel:
            if unique >= elem[0] :
                numericDataEncoded.append(0)
            else:
                numericDataEncoded.append(1)

        allPossibles.append(numericDataEncoded)

    gainIDX = Gain(numpy.transpose(numpy.array(allPossibles)),numericDataWithLabel[:,1])
    return (allPossibles[gainIDX])


def BuildModel(location):

    numericLoc = NumericLocation(location)
    f = open(location+'/dataset.data')
    attributes = f.read()

    attributes = attributes.split("\n")
    attributesList = [attribute.split(',') for attribute in attributes if not attribute.split(',').__contains__('?')]
    array = (numpy.array(attributesList))
    X = array[:, :-1]
    Y = array[:, -1]

    X_en, Y_en, X_encoders, Y_encoders = Encode(X, Y,numericLoc)

    X_train, X_test, y_train, y_test = train_test_split(X_en, Y_en, test_size=.15, random_state=45)

    pruningThreshold = 5/100 * len(attributesList)



    rootNode = DecisionTree(X_train, y_train, X_encoders, Y_encoders, list(range(numpy.shape(X_train)[1])),pruningThreshold,numericLoc)
    acc = Accuracy(rootNode,X_test, y_test)
    print(acc)

def NumericLocation(l):
    f = open(l + '/labelinfo.data').read()
    cls = ''
    dataCategory = []
    attribNames = []

    labels = f.split('\n')
    for g in labels[:-1]:
        if g.__contains__('class('):
            cls = g.split('(')[1][:-1]
        else:
            attrib = (g.split(' '))
            dataCategory.append(attrib[1])
            attribNames.append(attrib[0])

    category = numpy.array(list(map(int, dataCategory)))
    loc = [index for index, value in enumerate(category) if value == 1]
    return loc

if __name__ == '__main__':

    locations = ['Car Dataset','cylinder-bands','iris','mushroom']

    for location in locations[:]:
        BuildModel(location)
