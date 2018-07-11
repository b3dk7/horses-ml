import numpy as np
import tflearn

data, labels = tflearn.data_utils.load_csv("horse.csv")

def fix_data(data):
    for i in range(len(data)):
        if data[i][0] == "young":
            data[i][0] = 0
        else:
            data[i][0] = 1
    return data

data = fix_data(data)

def fix_labels(lables):
    for i in range(len(labels)):
        labels[i] = [i]
    return lables

labels= fix_labels(labels)


net = tflearn.input_data(shape=[None,6])
net = tflearn.fully_connected(net,10)
net = tflearn.fully_connected(net,1)
net = tflearn.regression(net, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)



m= tflearn.DNN(net)
m.fit(data, labels, n_epoch=100, batch_size=16, show_metric=True)

test, hope = tflearn.data_utils.load_csv("test-set.csv")

test = fix_data(test)

print("\n Input featrues: ", test)
print("\n Predicted output: ")
print(m.predict(test))
