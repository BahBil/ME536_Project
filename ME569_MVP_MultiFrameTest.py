import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import math
#from scipy.cluster.vq import vq, kmeans, whiten
import pandas as pd
#from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import mediapipe as mp
import cv2
from scipy import stats
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import normalize
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from math import sqrt
import gameplay as md
#from skimage import filters
#from skimage.filters import try_all_threshold
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

TotalData1 = np.empty([150,21,2])
RotatedData = np.empty([150,21,2])
ReshapedData1 = np.empty([150,42])
TotalData2 = np.empty([150,21,2])
#RotatedData2 = np.empty([150,21,2])
ReshapedData2 = np.empty([150,42])
TotalData3 = np.empty([750,21,2])
#RotatedData2 = np.empty([150,21,2])
ReshapedData3 = np.empty([750,42])
with open('objs5.pkl','rb') as f1: TotalData1 = pickle.load(f1)
with open('objs4.pkl','rb') as f2: TotalData2 = pickle.load(f2)
with open('ran.pkl','rb') as fran: RanData2 = pickle.load(fran)

f3 = open('objs.pkl','rb')
f4 = open('objs3.pkl','rb')
f5 = open('objs4.pkl','rb')
f6 = open('objs5.pkl','rb')
f7 = open('objs6.pkl','rb')

TotalData3 = np.vstack( (  pickle.load(f3) , pickle.load(f4) , pickle.load(f5) , pickle.load(f6) , pickle.load(f7)  ))

def ShapeIt(TotalData):
    RotatedData = np.empty(TotalData.shape)
    ReshapedData = np.empty([TotalData.shape[0],42])
    for i in range(TotalData.shape[0]):
        TotalData[i][:][0] =  TotalData[i][:][0] * 0.64*2
        TotalData[i][:][1] =  TotalData[i][:][1] * 0.48*2
        TotalData[i][:] = TotalData[i][:] - TotalData[i][0] 

        Angle = math.atan2(-TotalData[i][5][1], TotalData[i][5][0])
    
        for j in range(TotalData.shape[1]):
            RotatedData[i][j][0] =  math.cos(3.14/2-Angle) * TotalData[i][j][0]  + math.sin(3.14/2-Angle) * TotalData[i][j][1]
            RotatedData[i][j][1] =  math.sin(3.14/2-Angle) * TotalData[i][j][0]  - math.cos(3.14/2-Angle) * TotalData[i][j][1]

        RotatedData[i] = RotatedData[i] / RotatedData[i][5][1]
        if RotatedData[i][6][0] < RotatedData[i][5][0]:
            for j in range(TotalData.shape[1]):
                RotatedData[i][j][0] =  - RotatedData[i][j][0]

    #print(RotatedData[0])

    for i in range(TotalData.shape[0]):
        for j in range(TotalData.shape[1]):
            ReshapedData[i][j]    = RotatedData[i][j][0]
            ReshapedData[i][21+j] = RotatedData[i][j][1]
    return ReshapedData

ReshapedData2 = ShapeIt(TotalData2)
ReshapedData1 = ShapeIt(TotalData1)
ReshapedData3 = ShapeIt(TotalData3)
ReshapedRan = ShapeIt(RanData2) 

#print(ReshapedData)
#plt.imshow(ReshapedData)
#plt.colorbar()
#plt.show()

u,s,vt = np.linalg.svd(ReshapedData2.T,full_matrices=False)

#print(s)
#plt.plot(s,'r*')
#plt.show()


#rank = 2
#Mres = np.matmul(   u[:,:rank]   , np.matmul(np.diag(s[:rank]),vt[:rank,:])) 
"""
EnerPerc = sum(s[:rank]**2) / sum(s**2)
print(EnerPerc)

fig, axs = plt.subplots(1,2)
axs[0].imshow(ReshapedData)
axs[1].imshow(Mres)
#fig.colorbar()
plt.show()


whitened = whiten(Mres)
book = np.array( ( whitened[10],whitened[40],whitened[70],whitened[100],whitened[130] ) )
n_clusters = 5
kmean_model = KMeans(n_clusters=n_clusters)
#print(meansq[0])
centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_

"""
#clf =  svm.SVC()
#clf.fit(ReshapedData2, np.hstack((0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30))))[0,:])
#print(clf.predict(ReshapedData2))

#clf750 = LinearDiscriminantAnalysis(store_covariance=True)#KNeighborsClassifier()#DecisionTreeClassifier()#LinearDiscriminantAnalysis()#svm.SVC(probability=True)
#clf750.fit(np.vstack((ReshapedData3,ReshapedRan)), np.hstack((      0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                                                    0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
 #                                                                   0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
 #                                                                   0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                                                    0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
 #                                                                   np.array((range(5,35),))))[0,:])
#print(clf750.predict(ReshapedData3))
#print(clf750.score(ReshapedData3, np.hstack((   0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                        0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                        0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                        0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),
#                                        0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30))))[0,:]))

#scatter_matrix( meansq[0], alpha=0.2, figsize=(5, 5), diagonal='kde')
#clf751 = LinearDiscriminantAnalysis(store_covariance=True)#QuadraticDiscriminantAnalysis()#svm.SVC(probability=True)#
#clf751.fit(np.vstack((ReshapedData2,ReshapedRan)), np.hstack((0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30)),np.array((range(5,35),))))[0,:])
#print(clf751.predict(np.vstack((ReshapedData2,ReshapedRan))))

clf750 = LinearDiscriminantAnalysis(store_covariance=True)#QuadraticDiscriminantAnalysis()#svm.SVC(probability=True)#
TargetData = np.hstack((0*np.ones((1,30)),1*np.ones((1,30)),2*np.ones((1,30)),3*np.ones((1,30)),4*np.ones((1,30))))
clf750.fit(ReshapedData2, TargetData[0,:])
#print(clf750.predict(np.vstack((ReshapedData2))))



#
X_lda = clf750.transform(ReshapedData3)
#df = pd.DataFrame(ReshapedData2.T)

#corrMatrix = df.corr()
#plt.imshow(np.matmul(X_lda,X_lda.T)<600),



#CNN = Sequential()
#CNN.add(Dense(100,input_dim=42, activation='relu'))
#CNN.add(Dense(100, activation='relu'))
#CNN.add(Dense(35, activation='relu'))
#CNN.add(Dense(1, activation='sigmoid'))
CNN = tensorflow.keras.models.load_model("my_model")
CNN.summary()
#CNNTarget = np.hstack((  1*np.ones((1,750)),0*np.ones((1,30)) ))
CNNTarget = np.hstack((  1*np.ones((1,150)),0*np.ones((1,30)) ))
#CNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight=np.hstack((  1*np.ones((1,750)),25*np.ones((1,30)) )).tolist())
#CNN.fit(np.vstack((ReshapedData3,ReshapedRan)), CNNTarget.T ,epochs=100 ,batch_size=30)
#print(CNN.predict(np.vstack((ReshapedData3,ReshapedRan))))
#CNN.save("my_model")

print("Hello. I am a rock-paper-scissors-spock-reptile bot. try to win against me over 60 matches!")


"""
M=Mres
D = np.shape(M)[0]
N = np.shape(M)[1]
A = []                                                       
                                               #Sometimes the randomly picked A creates a "bad" similarty matrix that is hard to cluster (for this code)

r = np.linalg.matrix_rank(M)                                #step 1:  Find rank of datamatrix   
while np.linalg.matrix_rank(A) != r:                        #step 2-5: Repeat until rank of matrix A is equal to rank of matrix M  
    vcol = np.random.choice(N,size=r,replace=False)           #step 3:  Pick two index vectors  
    vcol = np.sort(vcol)                                      #step 3:  Pick two index vectors  
    vrow = np.random.choice(D,size=r,replace=False)           #step 3:  Pick two index vectors  
    vrow = np.sort(vrow)                                      #step 3:  Pick two index vectors  
    A=M[vrow,]                                                #step 4:  Construct A by slicing M
    A=A[:,vcol,]                                              #step 4:  Construct A by slicing M 
R=M[vrow,]                                                  #step 6:  Construct R by slicing M
P= np.matmul(np.linalg.inv(A),R)                            #step 7:  Construct P=A^-1 R
Q=np.matmul(np.transpose(P),P)                              #step 8:  Construct Q=P^T P
Q=np.absolute(Q)                                            #step 8:  Get absolute of Q   
w = np.linalg.matrix_power(Q, r)                            #step 9:  Get the similarity matrix      

fig, ax = try_all_threshold(np.matmul(vt.T,vt), figsize=(10, 8), verbose=False)
plt.show()
"""
#ReshapedData1 = pd.DataFrame(ReshapedData1,  index=np.hstack((0*np.ones(30),1*np.ones(30),2*np.ones(30),3*np.ones(30),4*np.ones(30))) ,  columns=np.array(range(42) ))
#ReshapedData = ReshapedData.drop(labels=['Channel', 'Region'], axis=1)
#T = preprocessing.Normalizer().fit_transform(ReshapedData1)

#change n_clusters to 2, 3 and 4 etc. to see the output patterns
#n_clusters = 150 # number of cluster

# Clustering using KMeans
#kmean_model = KMeans(n_clusters=n_clusters)
#kmean_model.fit(T)
#print(kmean_model.predict(T))
#centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
#print(centroids)
#print("labels")
#print(stats.mode(labels[0:30])[0])
#print(stats.mode(labels[30:60])[0])
#print(stats.mode(labels[60:90])[0])
#print(stats.mode(labels[90:120])[0])
#print(stats.mode(labels[120:150])[0])

# Dimesionality reduction to 2
#pca_model = PCA(n_components=2)
#pca_model.fit(T) # fit the model
#T = pca_model.transform(T) # transform the 'normalized model'
# transform the 'centroids of KMean'
#centroid_pca = pca_model.transform(centroids)
# print(centroid_pca)

# colors for plotting
#colors = ['blue', 'red', 'green', 'orange', 'magenta']
# assign a color to each features (note that we are using features as target)
#features_colors = [ colors[labels[i]] for i in range(len(T)) ]

# plot the PCA components
#plt.scatter(T[:, 0], T[:, 1],c=features_colors, marker='o',alpha=0)

# plot the centroids
#plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1],marker='x', s=100,linewidths=3, c=colors)

# store the values of PCA component in variable: for easy writing
#xvector = pca_model.components_[0] * max(T[:,0])
#yvector = pca_model.components_[1] * max(T[:,1])
#columns = ReshapedData1.columns
"""
# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.0005,
                head_width=0.02, alpha=0.75
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)
"""
plt.show()

#print(labels[0:30])
#print(labels[30:60])
#print(labels[60:90])
#print(labels[90:120])
#print(labels[120:150])



mp_drawing = mp.python.solutions.drawing_utils
mp_hands = mp.python.solutions.hands
RL_HandData = np.empty([21,2])
RL_RotatedData =  np.empty([21,2])
RL_ReshapedData =  np.empty([42])
# For webcam input:
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

OneImageTaken = False

Game1 = md.Game()
GamePlayerSet=()
PlayerHistory=()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
  

    
    key = cv2.waitKey(1)
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key%256 == 99:
        
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks( image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for indexxx , handtips in  enumerate(mp_hands.HandLandmark):
                RL_HandData[indexxx,0] = hand_landmarks.landmark[handtips].x * 0.64*2
                RL_HandData[indexxx,1] = hand_landmarks.landmark[handtips].y * 0.48*2

    
            RL_HandData[:] = RL_HandData[:] - RL_HandData[0] 
            Angle = math.atan2(-RL_HandData[5][1], RL_HandData[5][0])

            for j in range(RL_HandData.shape[0]):
                RL_RotatedData[j][0] =  math.cos(3.14/2-Angle) * RL_HandData[j][0]  + math.sin(3.14/2-Angle) * RL_HandData[j][1]
                RL_RotatedData[j][1] =  math.sin(3.14/2-Angle) * RL_HandData[j][0]  - math.cos(3.14/2-Angle) * RL_HandData[j][1]
    
    
            #image = cv2.putText(image, "x  " + str(RL_HandData[17][0])  , (10,320) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "y " + str(-RL_HandData[17][1]) , (10,360) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)

            #image = cv2.putText(image, "x  " + str(RL_RotatedData[17][0])  , (10,400) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "y " + str(RL_RotatedData[17][1]) , (10,440) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
    
            RL_RotatedData = RL_RotatedData / RL_RotatedData[5][1]
    
            if RL_RotatedData[6][0] < RL_RotatedData[5][0]:
                for j in range(RL_RotatedData.shape[0]):
                    RL_RotatedData[j][0] =  - RL_RotatedData[j][0]
                image = cv2.putText(image, "flipped"  , (400,40) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
    
        
            RL_ReshapedData[0:21]  = RL_RotatedData[:,0]
            RL_ReshapedData[21:42] = RL_RotatedData[:,1]
    
            #bestcandidate = np.argmin(  np.array(  np.linalg.norm(centroids[:]-RL_ReshapedData,axis=1) ))
    
            #if(bestcandidate == stats.mode(labels[0:30])[0]): bestcandidate = "Scissors"
            #if(bestcandidate == stats.mode(labels[30:60])[0]): bestcandidate = "Rock"
            #if(bestcandidate == stats.mode(labels[60:90])[0]): bestcandidate = "Paper"
            #if(bestcandidate == stats.mode(labels[90:120])[0]): bestcandidate = "Spock"
            #if(bestcandidate == stats.mode(labels[120:150])[0]): bestcandidate = "Reptile"
    
            #image = cv2.putText(image, str(bestcandidate)  , (10,270) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
            #image = cv2.putText(image, str(bestcandidate)  , (10,270) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
            #if(bestcandidate in (labels[0:30])): bestcandidate = "Scissors"
            #if(bestcandidate in (labels[30:60])): bestcandidate = "Rock"
            #if(bestcandidate in (labels[60:90])): bestcandidate = "Paper"
            #if(bestcandidate in (labels[90:120])): bestcandidate = "Spock"
            #if(bestcandidate in (labels[120:150])): bestcandidate = "Reptile"

            #if(bestcandidate in (labels[0:30])): bestcandidate = "Scissors"
            #if(bestcandidate in (labels[30:60])): bestcandidate = "Rock"
            #if(bestcandidate in (labels[60:90])): bestcandidate = "Paper"
            #if(bestcandidate in (labels[90:120])): bestcandidate = "Spock"
            #if(bestcandidate in (labels[120:150])): bestcandidate = "Reptile"
    
            #if ( clf.predict(np.array((RL_ReshapedData,))) == 0): bestcandidateSCV = "Scissors"
            #if ( clf.predict(np.array((RL_ReshapedData,))) == 1): bestcandidateSCV = "Rock"
            #if ( clf.predict(np.array((RL_ReshapedData,))) == 2): bestcandidateSCV = "Paper"
            #if ( clf.predict(np.array((RL_ReshapedData,))) == 3): bestcandidateSCV = "Spock"
            #if ( clf.predict(np.array((RL_ReshapedData,))) == 4): bestcandidateSCV = "Reptile"

            HandCode = clf750.predict(np.array((RL_ReshapedData,)))
            if ( HandCode[0] < Game1.NoOfSigns ): bestcandidateSCV750 = Game1.Names[int(HandCode[0])]
            else: bestcandidateSCV750 = str(HandCode)
            OneImageTaken = True
            print(bestcandidateSCV750)
            #image = cv2.putText(image, "Scissors Prob: " + str(int(10000*clf750.predict_proba(np.array((RL_ReshapedData,)))[:,0]))  , (10,200) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Rock Prob: " + str(int(10000*clf750.predict_proba(np.array((RL_ReshapedData,)))[:,1]))  , (10,220) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Paper Prob: " + str(int(10000*clf750.predict_proba(np.array((RL_ReshapedData,)))[:,2]))  , (10,240) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Spock Prob: " + str(int(10000*clf750.predict_proba(np.array((RL_ReshapedData,)))[:,3]))  , (10,260) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Reptile Prob: " + str(int(10000*clf750.predict_proba(np.array((RL_ReshapedData,)))[:,4]))  , (10,280) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
            #HandCode2 = clf751.predict(np.array((RL_ReshapedData,)))
            #if ( HandCode2[0] < Game1.NoOfSigns ): bestcandidateSCV751 = Game1.Names[int(HandCode2[0])]
            #else: bestcandidateSCV751 = str(HandCode2)
            #OneImageTaken = True

            #image = cv2.putText(image, "SVC_751: " + bestcandidateSCV751  , (10,175) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
            #image = cv2.putText(image, "Scissors Prob: " + str(int(10000*clf751.predict_proba(np.array((RL_ReshapedData,)))[:,0]))  , (10,300) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Rock Prob: " + str(int(10000*clf751.predict_proba(np.array((RL_ReshapedData,)))[:,1]))  , (10,320) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Paper Prob: " + str(int(10000*clf751.predict_proba(np.array((RL_ReshapedData,)))[:,2]))  , (10,340) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            ##image = cv2.putText(image, "Spock Prob: " + str(int(10000*clf751.predict_proba(np.array((RL_ReshapedData,)))[:,3]))  , (10,360) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #image = cv2.putText(image, "Reptile Prob: " + str(int(10000*clf751.predict_proba(np.array((RL_ReshapedData,)))[:,4]))  , (10,380) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
            NewHandProba = 100*CNN.predict(np.array((RL_ReshapedData,)),2)
            print(NewHandProba)


            if NewHandProba < 90:
                HandCode[0] = Game1.NoOfSigns
                Game1.Expand()
                
                print(np.argmax(np.dot(ReshapedRan,np.array((RL_ReshapedData,)).T).T/np.linalg.norm(ReshapedRan,axis=1)))
                ReshapedRan = np.delete(ReshapedRan,np.argmax(np.dot(ReshapedRan,np.array((RL_ReshapedData,)).T).T/np.linalg.norm(ReshapedRan,axis=1)),axis=0)   
                #print(CNNTarget.shape)
                CNNTarget=np.array((np.delete(CNNTarget,150),))
                #print(CNNTarget.shape) 

            if HandCode[0] >= 5:
                ReshapedData2 = np.vstack((ReshapedData2,RL_ReshapedData))

                for k in range(5):
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                        continue

                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                    image.flags.writeable = False
                    results = hands.process(image)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:

                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks( image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        for indexxx , handtips in  enumerate(mp_hands.HandLandmark):
                            RL_HandData[indexxx,0] = hand_landmarks.landmark[handtips].x * 0.64*2
                            RL_HandData[indexxx,1] = hand_landmarks.landmark[handtips].y * 0.48*2

    
                        RL_HandData[:] = RL_HandData[:] - RL_HandData[0] 
                        Angle = math.atan2(-RL_HandData[5][1], RL_HandData[5][0])

                        for j in range(RL_HandData.shape[0]):
                            RL_RotatedData[j][0] =  math.cos(3.14/2-Angle) * RL_HandData[j][0]  + math.sin(3.14/2-Angle) * RL_HandData[j][1]
                            RL_RotatedData[j][1] =  math.sin(3.14/2-Angle) * RL_HandData[j][0]  - math.cos(3.14/2-Angle) * RL_HandData[j][1]
    
    
                        #image = cv2.putText(image, "x  " + str(RL_HandData[17][0])  , (10,320) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
                        #image = cv2.putText(image, "y " + str(-RL_HandData[17][1]) , (10,360) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)

                        #image = cv2.putText(image, "x  " + str(RL_RotatedData[17][0])  , (10,400) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
                        #image = cv2.putText(image, "y " + str(RL_RotatedData[17][1]) , (10,440) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
    
                        RL_RotatedData = RL_RotatedData / RL_RotatedData[5][1]
    
                        if RL_RotatedData[6][0] < RL_RotatedData[5][0]:
                            for j in range(RL_RotatedData.shape[0]):
                                RL_RotatedData[j][0] =  - RL_RotatedData[j][0]
                            image = cv2.putText(image, "flipped"  , (400,40) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
    
        
                        RL_ReshapedData[0:21]  = RL_RotatedData[:,0]
                        RL_ReshapedData[21:42] = RL_RotatedData[:,1]

                        ReshapedData2 = np.vstack((ReshapedData2,RL_ReshapedData))
                #print(TargetData)
                #print(HandCode)
                TargetData = np.hstack((TargetData,HandCode[0]*np.ones((1,6))))
                #print(ReshapedData2.shape)
                #print(ReshapedRan.shape)
                #print(CNNTarget.shape)
                clf750.fit(ReshapedData2, TargetData[0,:])
                CNNTarget = np.hstack((  CNNTarget,1*np.ones((1,6)) ))

                CNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight=np.hstack((  1*np.ones((1,150)),5*np.ones(CNNTarget.shape) )).tolist())
                CNN.fit(np.vstack((ReshapedData2,ReshapedRan)), CNNTarget.T ,epochs=50 ,batch_size=150)
                print(CNN.predict(np.vstack((ReshapedData2,ReshapedRan))))
                HandCode = clf750.predict(np.array((RL_ReshapedData,)))
                if ( HandCode[0] < Game1.NoOfSigns ): bestcandidateSCV750 = Game1.Names[int(HandCode[0])]
                else: bestcandidateSCV750 = str(HandCode)
            
            image = cv2.putText(image, "SVC_750: " + bestcandidateSCV750  , (10,150) ,cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
            image = cv2.putText(image, "CNN Prob: " + str(np.around(NewHandProba))  , (10,200) ,cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
          
            lastimage = image
            print("Data Taken. Found a " + bestcandidateSCV750)
            
            Game1.PlayerResponse(int(HandCode[0]))

    if OneImageTaken:
        cv2.imshow('MediaPipe Hands', lastimage)
    else:
        cv2.imshow('MediaPipe Hands', image)


hands.close()
cap.release()