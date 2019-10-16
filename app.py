from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from keras.preprocessing import image
import cv2
import os
from sklearn.externals import joblib
from skimage.feature import hog
from flask import Flask,abort,render_template,request,redirect,url_for
from werkzeug import secure_filename

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')

def home():
	return render_template('home.html')

#@app.route('/upload_image' , methods= ['POST', 'GET'])


#def upload_file():
 #   if request.method =='POST':
  #      files = request.files.getlist('file[]',None)
   #        for file in files:
    #            filename = secure_filename(file.filename)
     #           file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
      #      return hello()
    #return render_template('file_upload.html')


@app.route('/predict',methods=['POST'])

def predict():
	#data = pd.read_csv("spam.csv", encoding="latin-1")

	full_filename = []
    
	with open("major_project_CNN_invert_nor.pkl", "rb") as file_handler:
		loaded_pickle = pickle.load(file_handler)
    


	if request.method == 'POST':
		sample = request.form['message']
		

	im = cv2.imread(sample)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im_gray /= 255
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	predict_ = 0
    # Threshold the image
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
	ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]




	number = []
	point_ = []
	for rect in rects:
        # Draw the rectangles
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    #     print (pt2)
		point_.append(pt2)
        # Resize the image
		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        
		roi = roi.reshape(1,28,28,1)
    #     roi = roi/255
		roi = 255- roi 
		roi = roi/255
        
		nbr = loaded_pickle.predict(roi)
        
        
		cv2.putText(im, str(int(np.argmax(nbr))), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        

		number.append((np.argmax(nbr)))

		def selection_sort(a):
			for i in range(len(a)):
				swap = i + np.argmin(a[i:,0])
				(a[i,0], a[swap,0]) = (a[swap,0], a[i,0])
				(a[i,1], a[swap,1]) = (a[swap,1], a[i,1])
			return a[:,1]

		a = np.array([point_, number])
		a = a.T

		user_num = selection_sort(a)

		numb = ''.join(map(str, user_num))
		numb = int(numb)

	cv2.imwrite(os.path.join('static/uploads', 'your_image__.jpg'), im)
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'your_image__.jpg')
	K.clear_session()

	#print( 'Above image seems to be a '+ prediction + ' image')
	return render_template('result.html',prediction = numb,
		user_image = full_filename)




if __name__ == '__main__':
	app.run(debug=True)

