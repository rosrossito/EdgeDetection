import pandas as pd
import numpy as np

prediction1 = pd.read_csv("input/results/prediction1.csv")
prediction2 = pd.read_csv("input/results/prediction2.csv")
prediction3 = pd.read_csv("input/results/prediction3.csv")
print(prediction1.shape)
print(prediction2.shape)
print(prediction3.shape)
print(prediction3.head(3))
prediction = np.concatenate((prediction1, prediction2, prediction3), axis=0)
print (prediction.shape)
prediction = np.ravel(prediction[: , 0])
print (prediction.shape)

results = pd.Series(prediction,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("input/results/cnn_mnist_datagen.csv",index=False)
