from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image,ImageTk
from tkinter import *
import tkinter as tk
import argparse
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image,ImageTk
from tkinter import *
import tkinter as tk

from PIL import Image, ImageTk

emotion_dict = {0: "Angry  ",  1: "Happy  ", 2: "Neutral  ",
                3: "Sad  ", 4: "Surprised  " , 5:"Emotions  "}

emoji_dist = {0: "emojis/angry.png",  1: "emojis/happy.png",
              2: "emojis/neutral.png", 3: "emojis/sad.png", 4: "emojis/surpriced.png" ,5:"emojis/default.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0,1,2,3,4,5]
#  first function to open the web cam
def show_vid():
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier = load_model(r'Emotion_little_vgg.h5')
    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print("test")
    cap = cv2.VideoCapture(0)
    i=0
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []

        if i >4:
            i = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            i += 1
            # rect,face,image = face_detector(frame)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
                x=preds.argmax()
                show_vid2(x)
                label = class_labels[x]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('print line 196',i)
    cap.release()
    cv2.destroyAllWindows()

# *************************************************************************************

def show_vid2(x):
    frame2 = cv2.imread(emoji_dist[show_text[x]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[x]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)
# *****************************************************************
if __name__ == '__main__':
    root = tk.Tk()
    # img = ImageTk.PhotoImage(Image.open("4-emoties-3.gif"))
    # heading = Label(root, image=img, bg='black')
    # heading.pack()
    heading2 = Label(root, text="Picture To Emotions", pady=20, font=('arial', 45, 'bold'), bg='yellow', fg='black')
    heading2.pack()
    # lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=50)
    # CDCDCD
    lmain3 = tk.Label(master=root, bd=10, fg="black", bg='yellow')
    # lmain.pack(side=LEFT)
    # lmain.place(x=50, y=250)q
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Picture To Emotions")
    root.geometry("1300x1000")
    root['bg'] = 'yellow'
    exitbutton = Button(root, text='  Quit   ', fg="black",bg='white', command=root.destroy, font=('arial', 18, 'bold')).pack(side=BOTTOM)
    btn = Button(root, text=' Open Camera '  ,fg='black', command=show_vid, bg='white', font=('arial', 20, 'bold'))
    btn.pack()
    # show_vid()
    show_vid2(-1)
    # btn2 = Button(root, text=' show Camera ', fg='black', command=lambda :show_vid2(2), bg='white', font=('arial', 20, 'bold'))
    # btn2.pack()
    root.update()
    root.mainloop()