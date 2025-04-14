from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import cv2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNet
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from PIL import Image, ImageTk

main = tkinter.Tk()
main.title("Optimizing Agriculture using ML")
main.geometry("1000x650")
# Load the image and resize it to fit the window
image = Image.open("front.png")
image = image.resize((1000, 650), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(image)

# Create a label to hold the background image
background_label = tkinter.Label(main, image=bg_image)
background_label.place(relwidth=1, relheight=1)

global cnn_model
global filename
#global X, Y
#global X_train, X_test, y_train, y_test, testImage, pca
global testImage, pca
global filename1
global X1,Y1
accuracy = []
precision = []
recall = []
fscore = []
global X1_train, X1_test, y1_train, y1_test
global cnn
global labels_Soil



with open('model/model1.json', "r") as json_file:
    loaded_model_json = json_file.read()
    cnn_classifier = model_from_json(loaded_model_json)
json_file.close()    
cnn_classifier.load_weights("model/model_weights1.h5")
cnn_classifier._make_predict_function() 


labels_Soil = ['Alluvial soil', 'Black soil','Clay soil', 'Red soil']
crops = ['Rice,Wheat, Sugarcane,Maize,Cotton', 'Virginia, Wheat,Jowar,Millets','Rice,Broccoli,Cabbage', 'Cotton,Pulses,Potatoes,OilSeeds']


low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])
leaf_labels = ['Apple___Apple_scab:drug1', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
              'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
              'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

leaf_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 
               'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
               'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

fertilizers = ['Mancozeb or Captan ', 'Copper-based fungicide', 'Sulfur or Lime Sulfur ', 
               'No fertilizer needed ', 'Azoxystrobin ', 'Mancozeb for Common Rust', 
               'No fertilizer needed', 'Chlorothalonil ', 'Copper-based fungicide ', 
               'Thiophanate-methyl ', 'No fertilizer needed ', 'Mancozeb ', 
               'Copper oxychloride ', 'No fertilizer needed ', 
               'Metalaxyl or Mancozeb ', 'Copper-based fungicide ', 
               'Chlorothalonil ', 'No fertilizer needed ', 'Metalaxyl or Chlorothalonil ', 
               'Chlorothalonil ', 'Mancozeb for Septoria ', 
               'Insecticidal soap or Abamectin', 'Difenoconazole', 
               'Copper-based fungicide', 'Imidacloprid or Acetamiprid']

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): #texture features will be extracted using histogram
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels_Soil)):
        if labels_Soil[i] == name:
            index = i
            break
    return index 

def uploadDataset1():
    global filename1
    filename1 = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename1+' Loaded\n\n')    
    text.insert(END,"Different Soil types Found in Dataset : "+str(labels_Soil)+"\n\n") 
    text.insert(END,"Total Soil types are : "+str(len(labels_Soil)))

def featuresExtraction():
    global filename1
    global X1,Y1
    global X1_train, X1_test, y1_train, y1_test
    text.delete('1.0', END)
    if os.path.exists("model/X1.npy"):
        X1 = np.load('model/X1.npy')
        Y1 = np.load('model/Y1.npy')
    else:
        X1 = []
        Y1 = []
        for root, dirs, directory in os.walk(filename1):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y1.append(class_label)
                    X1.append(features)
                    print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))
        X1 = np.asarray(X1)
        Y1 = np.asarray(Y1)
        np.save("model/X1",X1)
        np.save("model/Y1",Y1)
    X1 = X1.astype('float32')
    X1 = X1/255 #features normalization
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2)
    text.insert(END,"Extracted GLCM & Texture Features : "+str(X1[0])+"\n\n")
    text.insert(END,"Total images found in dataset : "+str(X1.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    text.insert(END,"80% training images : "+str(X1_train.shape[0])+"\n\n")
    text.insert(END,"20% training images : "+str(X1_test.shape[0])+"\n\n")

def calculateMetrics1(algorithm, predict, y1_test):
    a = accuracy_score(y1_test,predict)*100
    p = precision_score(y1_test, predict,average='macro') * 100
    r = recall_score(y1_test, predict,average='macro') * 100
    f = f1_score(y1_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    #conf_matrix = confusion_matrix(y1_test, predict) 
    #plt.figure(figsize =(6, 3)) 
    # Plot the confusion matrix

    conf_matrix = confusion_matrix(y1_test, predict) #calculate confusion matrix
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels_Soil, yticklabels = labels_Soil, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels_Soil)])

    print(labels_Soil)
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show() 

def runCNN():
    featuresExtraction()

    global X1_train, X1_test, y1_train, y1_test, X1, Y1, cnn
    global accuracy, precision,recall, fscore
    Y1 = to_categorical(Y1)
    XX = np.reshape(X1, (X1.shape[0], X1.shape[1], 1, 1))
    X1_train, X1_test, y1_train, y1_test = train_test_split(XX, Y1, test_size=0.2)
    if os.path.exists('model/model1.json'):
        with open('model/model1.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()    
        cnn.load_weights("model/model_weights1.h5")
        cnn._make_predict_function()   
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
        cnn.add(VGG16(weights='imagenet', include_top=True))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(XX, Y1, batch_size=12, epochs=10, shuffle=True, verbose=2)
        cnn.save_weights('model/model_weights1.h5')            
        model_json = cnn.to_json()
        with open("model/model1.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history1.pckl', 'wb')
        pickle.dump(hist.history1, f)
        f.close()
    print(cnn.summary())
    predict1 = cnn.predict(X1_test)
    predict1 = np.argmax(predict1, axis=1)
    y1_test = np.argmax(y1_test, axis=1)
    calculateMetrics1("Proposed CNN", predict1, y1_test)



def predict1():
    global cnn
    filename2 = filedialog.askopenfilename(initialdir="testSoil")
    img = cv2.imread(filename2)
    test = []
    img = cv2.resize(img, (64,64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = cnn.predict(test)
    predict = np.argmax(predict)

    img = cv2.imread(filename2)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Soil type Predicted as : '+labels_Soil[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Crop Recommended is : '+crops[predict], (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Soil Group Predicted as : '+labels_Soil[predict], img)
    cv2.waitKey(0)




def loadDataset():
    global filename, dataset,X, Y, testImage,pca
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" loaded\n\n")

    #global X, Y, testImage
    #text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    testImage = X[2].reshape(64,64,3)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Image Processing Completed\n\n")
    text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n")
    #text.delete('1.0', END)
    #global X, Y, testImage, pca
    text.insert(END,"Total features available in image before applying Features Extraction Algorithm: "+str(X.shape[1])+"\n") 
    if os.path.exists('model/pca.txt'):
        with open('model/pca.txt', 'rb') as file:
            pca = pickle.load(file)
            X = pca.fit_transform(X)
        file.close()
    else:
        pca = PCA(n_components = 1200)
        X = pca.fit_transform(X)
        with open('model/pca.txt', 'wb') as file:
            pickle.dump(pca, file)
        file.close()
    text.insert(END,"Total features available in image after applying Features Extraction Algorithm: "+str(X.shape[1])+"\n\n")
    text.update_idletasks()
    cv2.imshow("Segmented Image",cv2.resize(testImage,(300,300)))
    cv2.waitKey(0)

def trainCNN():
    text.delete('1.0', END)
    global X, Y
    global cnn_model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split for CNN training\n")
    text.insert(END,"80% dataset will be used for training and 20% for testing\n\n")
    text.insert(END,"Training Size: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Size: "+str(X_test.shape[0])+"\n")
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_model = model_from_json(loaded_model_json)
        json_file.close()
        cnn_model.load_weights("model/model_weights.h5")
        cnn_model._make_predict_function()   
    else:
        #creating cnn object
        cnn_model = Sequential()
        #defining layers of CNN
        cnn_model.MobileNet(weights='imagenet')
        cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(512))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(y_train.shape[1]))
        cnn_model.add(Activation('softmax'))
        #compiling the model
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start training model
        hist = cnn_model.fit(X_train, y_train, batch_size=16,epochs=15, validation_data=(X_test, y_test))
        cnn_model.save_weights('model/model_weights.h5')            
        model_json = cnn_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    #print(cnn_model.summary())
    #perfrom prediction on test data
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    #calculate accuracy score
    text.insert(END,'CNN Accuracy  : '+str(a)+"\n")
    text.insert(END,'CNN Precision : '+str(p)+"\n")
    text.insert(END,'CNN Recall    : '+str(r)+"\n")
    text.insert(END,'CNN FScore    : '+str(f)+"\n\n")
    text.update_idletasks()
    LABELS = leaf_labels 
    conf_matrix = confusion_matrix(y_test, predict) #calculate confusion matrix
    plt.figure(figsize =(16, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(LABELS)])
    plt.title("CNN Leaf Disease Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
        
def classification():
    global cnn_model, pca
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename) #read input image
    Z = np.float32(img.reshape((-1,3))) #create Z value from image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #create labels and centroids
    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels] #segment image based on labels
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC) #resize image
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, low_green, high_green) #remove out green part from the image so we have only infected part
    mask = 255-mask #masking done here
    res = cv2.bitwise_and(img, img, mask=mask) #extract infected part
    segmented = res #get segmented image
    res = res.ravel()
    test = []
    test.append(res)
    test = np.asarray(test)
    test = pca.transform(test) #extract features using PCA
    print(test.shape)
    test = test.astype('float32') #normalized the pixel values
    test = test/255
    preds = cnn_model.predict(test)#predict the disease using cnn model
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    text.insert(END,'Leaf Disease Detected & Classified as : '+leaf_labels[predict]+"\n")
    text.update_idletasks()
    cv2.putText(img, 'Leaf Disease Detected & Classified as : '+leaf_labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.putText(img, 'Recommended Fertilizers are as : '+fertilizers[predict], (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

    cv2.imshow('Leaf Disease Detected & Classified as : '+leaf_labels[predict], img)
    cv2.imshow("Segmented Image",cv2.resize(segmented,(200,200)))
    cv2.waitKey(0)

def trainLR():
    global model_yield
    
# Load the dataset
    data = pd.read_csv('Agricultural_yield.csv')

# Drop rows with missing production values
    data.dropna(subset=['Production'], inplace=True)

# Encode categorical variables
    label_encoder_state = LabelEncoder()
    label_encoder_district = LabelEncoder()
    label_encoder_crop = LabelEncoder()
    label_encoder_season = LabelEncoder()

    data['State'] = label_encoder_state.fit_transform(data['State_Name'])
    data['District'] = label_encoder_district.fit_transform(data['District_Name'])
    data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])
    data['Season'] = label_encoder_season.fit_transform(data['Season'])

# Remove non-numeric values from target variable
    data_numeric = data.copy()  # Make a copy of the dataframe
    data_numeric['Production'] = pd.to_numeric(data_numeric['Production'], errors='coerce')
    data_numeric.dropna(subset=['Production'], inplace=True)

# Split data into features and target variable
    X = data_numeric[['State', 'District', 'Crop', 'Season', 'Area', 'Crop_Year']]
    y = data_numeric['Production']

# Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Linear Regression Model
    model_yield = LinearRegression()  # Linear Regression Regressor
    model_yield.fit(X_train, y_train)
    y_pred = model_yield.predict(X_test)

# Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    text.delete('1.0', END)
    text.insert(END, 'R² Score is obtained: {:.4f}\n'.format(r2))


def predict_yield():
    program_path = r'C:\Users\harik\OneDrive\Desktop\CropOptimization\gui_code.py'  # Use raw string for the path
    try:
        # Use subprocess to run the external program
        subprocess.run(['python', program_path], check=True)  # Specify the python interpreter
        text.insert(END, 'Results obtained\n')
    except Exception as e:
        text.insert(END, f'Error obtained: {str(e)}\n')  # Include the error message for debugging


    

font = ('times', 15, 'bold')
font1 = ('times', 12, 'bold')

loadsoilButton = Button(main, text="Upload Soil Image Dataset", command=uploadDataset1)
loadsoilButton.place(x=15,y=150)
loadsoilButton.config(font=font1)  

cnnsoilButton = Button(main, text="Train CNN Algorithm", command=runCNN)
cnnsoilButton.place(x=15,y=200)
cnnsoilButton.config(font=font1)

clssoilButton = Button(main, text="Predict Soil and Recommend Plant", command=predict1)
clssoilButton.place(x=15,y=250)
clssoilButton.config(font=font1)

loadButton = Button(main, text="Upload Leaf Disease Dataset", command=loadDataset)
loadButton.place(x=15,y=300)
loadButton.config(font=font1)  

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=15,y=350)
cnnButton.config(font=font1)

clsButton = Button(main, text="Disease Classification", command=classification)
clsButton.place(x=15,y=400)
clsButton.config(font=font1)


cnnButton = Button(main, text="LR for Crop yield", command=trainLR)
cnnButton.place(x=15,y=450)
cnnButton.config(font=font1)

clsButton = Button(main, text="Predict Yield", command=predict_yield)
clsButton.place(x=15,y=500)
clsButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=250,y=150)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
