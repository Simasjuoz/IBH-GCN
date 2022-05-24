from spektral.utils.sparse import sp_matrix_to_sp_tensor
from Config import *
import time
from GraphPrep import getAdjMatrix

#
tf.config.run_functions_eagerly(False)

################################################################################
# Config
################################################################################
learning_rate = LEARNING_RATE
training_logs = ""
################################################################################
# Load data
################################################################################

class GetDataset(Dataset):
    def __init__(self, data, pOfData = 1, rotated = False, testing = False, rotation_degree = 0, **kwargs):
        self.rotation_degree = rotation_degree
        self.rotated = rotated
        self.testing = testing
        self.data = data
        self.p = pOfData
        super().__init__(**kwargs)

    def read(self):
        def make_graph(a, image, label,debug=False):

            if debug:
                plt.imshow(image)
                plt.show()

            # Node features
            x = []
            if stride == block_size:
                x = splitImage(image)
            else:
                makePatches(image, 0, 0, x)

            x = np.array(x)        
            
            if debug:
                showImg(x)

            x = np.reshape(x, (block_amount**2,-1))

            for feat in range(len(x)):
                x[feat] = -np.sort(-x[feat])

            # Labels
            y = label

            return Graph(x=np.array(x), y=y)
        self.a = getAdjMatrix(block_amount)

        self.a = sp.csr_matrix(self.a)
        self.a = GCSConv.preprocess(self.a)
        (x_train, y_train), (x_test, y_test) = self.data
        x_train = x_train[0:int(x_train.shape[0]*self.p // 1)]
        y_train = y_train[0:int(y_train.shape[0]*self.p // 1)]
        x_test = x_test[0:int(x_test.shape[0]*self.p // 1)]
        y_test = y_test[0:int(y_test.shape[0]*self.p // 1)]
        if self.rotated:
            x_test = rotateImages(x_test, self.rotation_degree, self.testing)
            return [make_graph(self.a, x_test[_], y_test[_], debug=False) for _ in tqdm(range(x_test.shape[0]))]
        return [make_graph(self.a, x_train[_], y_train[_], debug=False) for _ in tqdm(range(x_train.shape[0]))]

################################################################################
# Build model
################################################################################
class Robin(Model):
    def __init__(self, denseNeu, outerNeu, outNeu):
        super().__init__()
        self.denseNeu = denseNeu*4
        self.dropoutLayer = Dropout(DROPOUT)
        self.dense1 = tf.keras.layers.Dense(denseNeu, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense2 = tf.keras.layers.Dense(denseNeu*2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense3 = tf.keras.layers.Dense(denseNeu*4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv1 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv2 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv3 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG)) #TODO i think we can just call conv1 5 times instead of defining 5 identical layers??? andemand was here
        self.conv4 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv5 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        
        self.global_pool = GlobalMaxPool()
        self.global_avg_pool = GlobalAvgPool()
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense = Dense(outNeu, activation="softmax")

    def call(self, inputs):
        x, a = inputs

        x = self.dense1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.dropoutLayer(x)
        x = self.dense3(x)
        x = self.dropoutLayer(x)

        x = tf.reshape(x, [-1, block_amount*block_amount, self.denseNeu])
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        x = self.conv5([x, a])

        output = self.global_avg_pool(x)
        output2 = self.global_pool(x)
        output = tf.concat([output, output2], 1)
        output = self.fc1(output)
        output = self.dropoutLayer(output)
        output = self.dense(output)
        return output

class IBH(Model):
    def __init__(self,  innerNeu, denseNeu, outerNeu, outNeu):
        self.blockAdj = getAdjMatrix(block_size)
        self.blockAdj = sp.csr_matrix(self.blockAdj)
        #self.blockAdj = GCSConv.preprocess(self.blockAdj)
        self.blockAdj = sp_matrix_to_sp_tensor(self.blockAdj)
        self.innerNeu = innerNeu
        self.denseNeu = denseNeu*4
        super().__init__()
        self.dropoutLayer = Dropout(DROPOUT)
        self.blockConv1 = GCSConv(innerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.blockConv2 = GCSConv(innerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.blockConv3 = GCSConv(innerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense1 = tf.keras.layers.Dense(denseNeu*1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense2 = tf.keras.layers.Dense(denseNeu*2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense3 = tf.keras.layers.Dense(denseNeu*4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv1 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv2 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv3 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG)) #TODO i think we can just call conv1 5 times instead of defining 5 identical layers??? andemand was here
        self.conv4 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.conv5 = GCSConv(outerNeu, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        
        self.global_pool = GlobalMaxPool()
        self.global_avg_pool = GlobalAvgPool()
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(L2REG))
        self.dense = Dense(outNeu, activation="softmax")

    def call(self, inputs):
        x, a = inputs
        x = tf.reshape(x, [-1, block_size**2, channel_amount])
        x = self.blockConv1([x, self.blockAdj])
        x = self.blockConv2([x, self.blockAdj])
        x = self.global_avg_pool(x)
        x = tf.reshape(x, [-1, block_amount**2, self.innerNeu])

        x = self.dense1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.dropoutLayer(x)
        x = self.dense3(x)
        x = self.dropoutLayer(x)

        x = tf.reshape(x, [-1, block_amount*block_amount, self.denseNeu])
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        x = self.conv5([x, a])
        output = self.global_avg_pool(x)
        output2 = self.global_pool(x)
        output = tf.concat([output, output2], 1)
        output = self.fc1(output)
        output = self.dropoutLayer(output)
        output = self.dense(output)
        return output


optimizer = Adam(learning_rate=LEARNING_RATE)
loss_fn = SparseCategoricalCrossentropy()


################################################################################
# Fit model
################################################################################
@tf.function(experimental_relax_shapes=True)
def train_step(model, inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(model, loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
            len(target), 
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

trainTimeArr = []

def train(model, loader_tr, loader_va):
    
    START_TIME = time.time()
    training_logs = ""
    epoch = step = 0
    best_val_loss = np.inf
    best_weights = None
    patience = ES_PATIENCE
    results = []
    prevTime = 0
    for batch in loader_tr:
        step += 1
        loss, acc = train_step(model,*batch)
        results.append((loss, acc))
        if step == loader_tr.steps_per_epoch:
            step = 0
            epoch += 1
 
            # Compute validation loss and accuracy
            val_loss, val_acc = evaluate(model, loader_va)
            print(
                "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                    epoch, *np.mean(results, 0), val_loss, val_acc
                )
            )
            training_logs += "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f} \n".format(
                    epoch, *np.mean(results, 0), val_loss, val_acc
                )
            timer = (time.time() - START_TIME)
            timePrE = timer - prevTime 
            prevTime += timePrE
            training_logs += "{:.3f} seconds\n".format(timePrE)
            trainTimeArr.append(timePrE)

            # Check if loss improved for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = ES_PATIENCE
                print("New best val_loss {:.3f}".format(val_loss))
                training_logs += "New best val_loss {:.3f} \n".format(val_loss)
                best_weights = model.get_weights()
                model.save_weights(model_path)
            else:
                patience -= 1
                if patience%5==0:
                    newLr = optimizer._decayed_lr(float)/2
                    optimizer._set_hyper("learning_rate", newLr)
                    print("NEW LR: " + str(optimizer._decayed_lr(float)))
                if patience == 0:
                    print("Early stopping (best val_loss: {})".format(best_val_loss))
                    break
            results = []
    return best_weights, training_logs
################################################################################
# Evaluate model
################################################################################

def testModelRand(model,data, training_logs="No train logs - only tested", numOfTesting=1, postFix = ""):
    model.load_weights(model_path)
    acc = []
    output = ""

    for i in range(numOfTesting):
        print("Testing.. rotation degree: " + str(i))
        rotatedData = GetDataset(data,1, rotated=True, testing=False, transforms=NormalizeAdj())
        loader_te = MixedLoader(rotatedData, batch_size=BATCH_SIZE, shuffle=False)
        test_loss, test_acc = evaluate(model, loader_te)
        acc.append(test_acc)
        output += "Iteration: {}. Test loss: {:.4f}. Test acc: {:.4f}".format(i, test_loss, test_acc)
        if test_acc > 0.5:
            output += " -- NICE!"
        output += "\n"
    
    output += "OA: " + str(np.mean(np.array(acc)))
    output += "\n"
    output += "BLOCKSIZE: " + str(block_size)
    output += "\n"
    output += "BLOCKAMOUNT: " + str(block_amount)
    output += "\n"
    output += "stride: " + str(stride)
    output += "\n"
    output += "DROPOUT CHANCE: " + str(DROPOUT)
    output += "\n"
    output += "L2REG: " + str(L2REG)
    output += "\n"
    output += str(np.mean(np.array(trainTimeArr)))

    output += "\n\n\n" + training_logs

    text_file = open("./Logs/" + model_used + model_name + postFix + "Logs.txt", "w")
    text_file.write(output)
    text_file.close()

    text_file = open("./Logs/" + model_used + model_name + postFix + "ModelSummary.txt", "w")
    model.summary(print_fn=lambda x: text_file.write(x + '\n'))
    text_file.close()

def testModel(model, data, postFix = ""):
    model.load_weights(model_path)
    acc = []
    output = ""
    for i in range(0,360,30):
        print("Testing.. rotation degree: " + str(i))
        rotatedData = GetDataset(data, rotated=True, testing=True, rotation_degree=i, transforms=NormalizeAdj())
        loader_te = MixedLoader(rotatedData, batch_size=BATCH_SIZE, shuffle=False)
        test_loss, test_acc = evaluate(model, loader_te)
        acc.append(test_acc)
        output += "Degree: {}. Test loss: {:.4f}. Test acc: {:.4f}".format(i, test_loss, test_acc)
        if test_acc > 0.5:
            output += " -- NICE!"
        output += "\n"
        if i == 330:
            output += "OA: " + str(np.mean(np.array(acc)))
            output += "\n"
            output += "BLOCKSIZE: " + str(block_size)
            output += "\n"
            output += "BLOCKAMOUNT: " + str(block_amount)
            output += "\n"
            output += "stride: " + str(stride)
            output += "\n"
            output += "DROPOUT CHANCE: " + str(DROPOUT)
            output += "\n"
            output += "L2REG: " + str(L2REG)
            output += "\n"

    output += "\n\n\n" + training_logs

    text_file = open("./Logs/" + model_used + "/" + model_name + postFix + "Logs.txt", "w")
    text_file.write(output)
    text_file.close()

    text_file = open("./Logs/" + model_used + "/" + model_name + postFix + "ModelSummary.txt", "w")
    model.summary(print_fn=lambda x: text_file.write(x + '\n'))
    text_file.close()