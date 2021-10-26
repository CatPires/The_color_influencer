from tensorflow import keras
model = keras.models.load_model('model3')
import cv2
from sklearn.cluster import KMeans
import numpy as np
import glob
#import matplotlib.pyplot as plt
from IPython.display import Image, display

def color_extractor_simple(path):
    #img_lst = 
    try: 
        image = cv2.imread(path,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.imshow(image)
        # reshape image
        pixels = image.reshape((-1,3)).astype("float32") / 255
        # Cluster the pixels
        centers = KMeans(n_clusters=10).fit(pixels).cluster_centers_
        res = (centers) #.astype("float64")
    except:
        print('Not_found')
    return res

def predic():
    #path_to_target = folder #'Behance\\' path_to_target +
    path_to_file_list = glob.glob('*.jpg')
    img = [f for f in path_to_file_list]
    img_array = []
    
    for file in img:
        colors = color_extractor_simple(file)
        img_array.append(colors)
        #display(Image(filename=file))
   
    imp = np.array(img_array)
    pred = model.predict(imp)  #.reshape(1,-1)[0]

    for i in range(len(img)):
        display(Image(filename=img[i]))
        print('%d likes predict to this photo' %(pred[i].item()))
        #print(pred[i])

    return #pred #.item()

predic()