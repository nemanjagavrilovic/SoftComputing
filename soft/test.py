import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD, Adagrad
from keras.utils import np_utils, to_categorical
from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier

#Kreiranje neuronske mreze
def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=1568, activation='sigmoid'))
    ann.add(Dense(5, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array (X_train, np.float32)  # dati ulazi
    y_train = np.array (y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD (lr=0.01, momentum=0.9)
    ann.compile (loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit (X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()
def dilate(image):
    kernel = np.ones((4,4)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=2)
def erode(image):
    kernel = np.ones((4,4)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=3)


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255



def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


#selektovanje regiona prilikom obucavanja
def select_region(image_bin):
    x = 1130;
    y = 1280;
    h = 340;
    w = 200;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    region=resize_region(region)
    region = scale_to_range(region)
    region=matrix_to_vector(region)
    return  region

#selektovanje regiona prilikom testiranja slike
def select_test_region(image_bin):
    x = 1812;
    y = 111;
    h = 125;
    w = 249;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    plt.imshow(region)
    plt.show()
    region=resize_region(region)
    region=scale_to_range(region)
    region=matrix_to_vector(region)
    return  region


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros ((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum (axis=1)
    rect[0] = pts[np.argmin (s)]
    rect[2] = pts[np.argmax (s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff (pts, axis=1)
    rect[1] = pts[np.argmin (diff)]
    rect[3] = pts[np.argmax (diff)]

    # return the ordered coordinates
    return rect

#izdvajanje konture karte i prikazivanje kao posebne slike
def obrada_ucenja(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    print("Warpovanje")
    rect = order_points (pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt (((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt (((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max (int (widthA), int (widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt (((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt (((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max (int (heightA), int (heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    w=int(maxWidth*7)
    h=int(maxHeight*4.5)
    dst = np.array ([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform (rect, dst)
    warped = cv2.warpPerspective (image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped,maxWidth,maxHeight


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    print("Warpovanje")
    #display_image(image)
    rect = order_points (pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt (((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt (((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max (int (widthA), int (widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt (((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt (((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max (int (heightA), int (heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array ([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform (rect, dst)
    warped = cv2.warpPerspective (image, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, (200, 200), interpolation=cv2.INTER_NEAREST)

    # return the warped image
    return warped,maxWidth,maxHeight

#izdvajanje karte prilikom obucavanja mreze
def select_card_contour(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array=[]
    pts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect (contour)
        rect,f,angle=cv2.minAreaRect(contour)
        cv2.rectangle (image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        size=cv2.contourArea(contour)
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.1*peri,True)
        #ukoliko je velicina konture veca od 150000 uzima se ta kontura jer
        #je to karta
        if(size>150000):
           #print (size)
           #print (len (approx))
           #print ('-------')
           list=[]

           for idx,i in enumerate(approx):
               list.append(approx[idx][0])
           pts=np.array(list,np.float32)
           regions_array.append(image_bin[y:y+h+1,x:x+w+1])


    #regions_array = sorted (regions_array, key=lambda item: item[1][0])
   # sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array,pts


#izdvajanje karata sa test slika
def select_testcards(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array=[]
    box = []
    pts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect (contour)
        rect,f,angle=cv2.minAreaRect(contour)
        #cv2.rectangle (image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        size=cv2.contourArea(contour)
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.01*peri,True)
        if(size>100000):
           #print (size)
           #print (len (approx))
           #print ('-------')
           list=[]

           for idx,i in enumerate(approx):
               list.append(approx[idx][0])
           pts=np.array(list,np.float32)
           #tacke uglova karte
           box.append(pts)
           #print("Iz vise slika")
           #display_image(image_bin[y:y+h+1,x:x+w+1])
           regions_array.append(image_bin[y:y+h+1,x:x+w+1])
           print(len(pts))


    #regions_array = sorted (regions_array, key=lambda item: item[1][0])
    #sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array,box



def select_sign_contour(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array=[]

    print("Broj kontura iz coska")
    print(len(contours))
    hull=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect (contour)
        rect,f,angle=cv2.minAreaRect(contour)
        #cv2.rectangle (image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        size=cv2.contourArea(contour)
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.01*peri,True)
        print("Iz konture")
        print(size)
        if(size>4000):
            hull = cv2.convexHull(contour, returnPoints=True)
            print (hull)
            cv2.drawContours(image_orig, [hull], -1, (255, 0, 0), 1)
            plt.imshow(image_orig)
            plt.show()

            #regions_array.append(image_bin[y:y+h+1,x:x+w+1])


    #regions_array = sorted (regions_array, key=lambda item: item[1][0])
    #sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, hull



def matrix_to_vector(image):
    return image.flatten()


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    return image/255


def convert_output(names):
    nn_outputs = []
    for index in range(len(names)):
        output = np.zeros(len(names))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def winner(output): # output je vektor sa izlaza neuronske mreze

    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor (image, cv2.COLOR_RGB2GRAY)
    #blur = cv2.GaussianBlur (gray, (5, 5), 0)
    gray= erode(dilate(gray))
    img_w, img_h = np.shape (image)[:2]
    bkg_level = gray[int (img_h / 100)][int (img_w / 2)]
    thresh_level = bkg_level + 60

    retval, thresh = cv2.threshold (gray, thresh_level, 255, cv2.THRESH_BINARY)
    obrada=erode(dilate(thresh))

    return thresh

def select_test_region(contour,w,h):
    #dimenzije konture znaka koje su nam potrebne
    x = 0
    y = 20
    viewh = 20;
    vieww = 27;
    if(w>h):

        print("Rotacija")
        Mat = cv2.getRotationMatrix2D((vieww / 2, viewh / 2), -90, 1)
        contour = cv2.warpAffine(contour, Mat, (vieww, viewh))
        region = contour[y:y + viewh + 1, x:x + vieww + 1]
        print("Iseceno")
        plt.imshow(region, 'gray')
        plt.show()

        return region
    else:
        print("Else")

        region = contour[y:y + viewh + 1, x:x + vieww + 1]
        region = cv2.resize(region, (225, 229), interpolation=cv2.INTER_NEAREST)

        print("Iseceno")
       # plt.imshow(region, 'gray')
        #plt.show()

    return region

def select_sign(contour):
    x = 0;
    y = 300;
    h = 230;
    w = 240;
    #dimenzije znaka prilikom obucavanja
    region = contour[y:y + h + 1, x:x + w + 1]
    #plt.imshow(region, 'gray')
    #plt.show()

    return region

def testiranje_rotacije():

    pictures=os.listdir('images')
    data = []
    labels = []
    for card in pictures:
        cardname = card.split('.')[0].split(' ')[0]
        labels.append (cardname)
        image_color = load_image ('images/' + card)
        image = preprocess_image (image_color)
        img, contours, box = select_card_contour(image_color.copy (), image)

        warp,w,h=obrada_ucenja(img,(box))
        region = select_sign(warp)
        slika_obradjena = invert(image_bin(image_gray(region)))
        img_bin = erode(dilate(slika_obradjena))
        img, c= select_sign_contour(region.copy(), slika_obradjena)
        print(len(c))
        region = resize_region(np.array(c,np.float32))
        region = scale_to_range(region)
        region = matrix_to_vector(region)

        data.append (region)

    te_labels=to_categorical(labels);
    ann = create_ann()
    print ("Mreza kreirana")

    #ann = train_ann (ann, data, te_labels)
   # ann.save ('NeuralNetwork.h5')
    test(labels,ann)


sign_name={1:"herc",2:"karo",3:"pik",4:"tref"}

def test(labels,ann):
    print("test")
    te_labels = convert_output(labels)

    ann=load_model('NeuralNetwork.h5')
    image_color = load_image('test/24.jpg')
    image = preprocess_image(image_color)
    # obrada=invert(image_bin(image_gray(image_color)))
    # kls = erode(dilate(obrada))
    img, contours, box = select_testcards(image_color.copy(), image)
    display_image(img)
    for i in range(len(contours)):
        warp, w, h = four_point_transform(img, (box[i]))
        display_image(warp)
        region = select_test_region(warp, w, h)
        slika_obradjena = invert(image_bin(image_gray(region)))
        # obradjena=preprocess_image(region)
        img_bin = erode(dilate(slika_obradjena))
        i, c = select_sign_contour(region.copy(), slika_obradjena)

        r = resize_region((np.array(c, np.float32)))
        r = scale_to_range(r)
        r = matrix_to_vector(r)
        input = [r]

        result = ann.predict(np.array(input),verbose=0)
        print (result)
        maxIndex = np.argmax(result)
        print(sign_name[maxIndex])
        #print (display_result(result, labels))


testiranje_rotacije()

def trainPokerNetwork():
    f = open('CardDataSet.txt','r')
    textLines = []
    labels = []
    ranks = []
    numbers = []
    for line in f:
        newData = []
        ranks = []
        numbers = []
        parts = line.split(",")
        ranks.append(parts[0])
        labels.append(parts[0])
        print(labels)
        numbers.append(parts[1])
        print(ranks)

        print(numbers)
#        ranks = sorted(ranks, key=int)
 #       numbers = sorted(numbers, key=int)
  #      highestNumber = numbers[-1]
   #     numbersExtracted = []
    #    numbersExtracted.append(highestNumber)
     #   ranksExtracted = []
       # for s in range(len(ranks)-1):
       #     diff = ranks[s+1] - ranks[s]
       #     ranksExtracted.append(diff)

      #  for n in range(len(numbers)-1):
      #      diff = numbers[n+1] - numbers[n]
      #      numbersExtracted.append(diff)

        newData.extend(ranks)
        newData.extend(numbers)
        textLines.append(newData)

    print("Priprema ulaza.....")
    #
    data = []
    for x in  textLines:
        data.append(x)
    #
    data = np.array(data)
    #
    print("Zavrsena priprema.....")
    #
    testLabels = np.array(labels)

    model = RandomForestClassifier(n_estimators=10,n_jobs=-1, criterion='entropy')
    model.fit(data, testLabels)
    #score = cross_val_score(model, data, testLabels)
    #print score.mean()
    print(model.score(data, testLabels))
    return model

#Klasifikator (predvidjanje iz fajla)
#trainNetwork()
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder

def klasifikacija():
    column = ["S1", "Y"]
    df = pd.read_csv('PokerHandDataset.txt')
    df.head()
    x = df.iloc[:,2:-1].copy()
    y=df.iloc[:,10].copy()
    enc = OneHotEncoder(sparse=False)
    enc.fit(x)
    dum = enc.transform(x)
    X = pd.DataFrame(dum)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y , test_size=0.3)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(clf.score(x_test, y_test))
