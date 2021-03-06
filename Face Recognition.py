import cv2, numpy, os
os.chdir(r'D:/')
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'FaceData'

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
			images.append(cv2.imread(path, 0))
			lables.append(int(lable))
		id += 1
width, height = 130, 100

# Create a Numpy array 
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# Trains Model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Face Recognition 
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
c=1
while c== 1 :
	(_, im) = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 4)
	print(faces)
	for (x, y, w, h) in faces:
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (width, height))
		# Try to recognize the face
		prediction = model.predict(face_resize)
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0),2)

		if prediction[1]<500:

		   cv2.putText(im, '% s' %(names[prediction[0]]), (x-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,255,0))
		else:
		   cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

	cv2.imshow('Recognizing', im)
	
	key = cv2.waitKey(10)
	if key == 27:
		break
