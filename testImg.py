from PIL import Image
import glob
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
from pathlib import Path

#accuracy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

trainedModel = "memory.pkl"

print("Ingrese ruta de la imagen de prueba")
testImageDir = input()

print("Ajustando imagen de prueba")
testImage = Image.open(testImageDir)
testRGB = np.array(testImage)
testRGB.ravel()
testRGB =testRGB.reshape(1,-1)

print("Verificando existencia de modelo entrenado")
modelPath = Path(trainedModel)
if modelPath.is_file():
    clf = joblib.load('memory.pkl')
else: 
    print("Modelo entrenado no encontrado")
    features = []
    labels = []

    print("Recopilando imagenes de entrenamiento")
    imageDir = glob.glob("TestImages/*/*.jpg")

    print("Ajustando imagenes de entrenamiento")
    for i in range(len(imageDir)):
        label = imageDir[i]
        label = label[label.find("\\")+1]
        trainImage = Image.open(imageDir[i])
        rgbImg = np.array(trainImage)
        rgbImg = rgbImg.ravel() 
        rgbImg = rgbImg.reshape(1,-1)
        features.extend(rgbImg)
        labels.extend(label)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
    print("Entrenando")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)

    prediction = clf.predict(x_test)
    print("Precisión del modelo: ",accuracy_score(y_test,prediction)*100,"%")

    print("Guardando modelo")
    joblib.dump(clf,trainedModel)

print("Clasificando imagen de prueba")
resultado = clf.predict(testRGB)
print(resultado)
