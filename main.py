from Model import *
from DatasetGetter import *

def GetDataSet(datasetName):
    return {
        DatasetEnum.SVHN.value: (getSvhn(), (64, 128, 9, 32)), #128, 256
        DatasetEnum.C10.value: (getCifar10(), (32, 64, 10, 32)),
        DatasetEnum.C100.value: (getCifar100(), (32, 64, 100, 32)),
        DatasetEnum.MNIST.value: (getMnist(), (8, 32, 10, 8))
    }[datasetName]

def GetModel(modelName, denseNeu, outerNeu, outNeu, innerNeu = 32):
    return {
        ModelnameEnum.HASBULLAH.value: Hasbullah(innerNeu, denseNeu, outerNeu, outNeu),
        ModelnameEnum.ROBIN.value: Robin(denseNeu, outerNeu, outNeu),
    }[modelName]

def processData(data, p):
    tr, te = data
    x, y = tr

    trShape = int((x.shape[0]*p)//1)
    vaShape = int(((trShape * 0.9) * p) // 1)

    graphData = GetDataset(data,p, transforms=NormalizeAdj())
    rotatedData = GetDataset(data,p, rotated=True, transforms=NormalizeAdj())

    data_tr = graphData[0:vaShape]
    data_va = graphData[vaShape:trShape]
    data_te = rotatedData

    loader_tr = MixedLoader(data_tr, batch_size=BATCH_SIZE, epochs=EPOCHS)
    loader_va = MixedLoader(data_va, batch_size=BATCH_SIZE)
    loader_te = MixedLoader(data_te, batch_size=BATCH_SIZE)

    return loader_tr, loader_va, loader_te

def trainOrTestModel(doTrain, doTest, doTest2, model, data, p, postfix):
    if doTrain and doTest:
        l_tr, l_va, l_te = processData(data, p)
        weights, logs = train(model, l_tr, l_va)
        testModelRand(model, data, logs, numOfTesting=1, postFix=postfix)
        testModel(model, data, postFix=postfix+"2")
        return
    if doTrain:
        l_tr, l_va, l_te = processData(data, p)
        weights, logs = train(model, l_tr, l_va)
    if doTest:
        testModelRand(model, data, numOfTesting=3, postFix=postfix)
    if doTest2:
        testModel(model, data, postFix=postfix+"2")
    

def main():
    data, neuAmount = GetDataSet(dataset_name)
    outerNeu, denseNeu, outNeu, innerNeu = neuAmount
    model = GetModel(model_used, outerNeu, denseNeu, outNeu, innerNeu)
    #model.load_weights(model_path)
    trainOrTestModel(True, True, True, model, data, 1, postfix=str(block_size) + "x" + str(block_size)+ "t10101")
main()
