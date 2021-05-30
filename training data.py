import matplotlib.pyplot as plt

from main import*
from sklearn.model_selection import train_test_split

path = 'AI CAR DATA SET'
data = importDataInfo(path)

balanceData(data,display=True)

imagesPath,steerings = loadData(path,data)
xTrain, xVal,yTrain,yVal = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total training images:',len(xTrain))
print('Total validation images:',len(xVal))

model=createModel()
model.summary()

history=model.fit(batchGen(xTrain,yTrain,100,3),steps_per_epoch=300,epochs=10,validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim(0,1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


