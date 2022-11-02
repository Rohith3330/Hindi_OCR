from flask import Flask,render_template,request,flash,redirect
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imshow
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
from keras.applications.imagenet_utils import decode_predictions


app = Flask(__name__)

model_path='models/model12.h5'

model = load_model(model_path)
model.make_predict_function()


di={'sp_char_27 ':'क्र','sp_char_1':'क्ल','character_3_ga':'ग​','sp_char_40':'च्य','character_36_gya':'ज्ञ','character_11_taamatar':'ट','character_8_ja':'ज','character_20_na':'न','sp_char_5':'ज्व','character_33_ha':'ह','character_19_dha':'ध','character_29_waw':'व','character_16_tabala':'त','character_7_chha':'छ​','character_24_bha':'भ​','sp_char_12':'ट्ट','sp_char_4':'ज्र','character_10_yna':'ञ','character_25_ma':'म','character_14_dhaa':'ढ​','character_15_adna':'ण','character_12_thaa':'ठ​','character_35_tra':'त्र​','character_32_patalosaw':'स','character_9_jha':'झ','character_6_cha':'च','character_4_gha':'घ​','character_23_ba':'ब','character_28_la':'ल','character_22_pha':'फ​','character_2_kha':'ख​','character_34_chhya':'क्ष','character_26_yaw':'य','character_27_ra':'र​','sp_char_36':'च्म','character_17_tha':'थ​','character_18_da':'द​','character_30_motosaw':'श','character_21_pa':'प','sp_char_37':'च्छ​','character_13_daa':'ड','character_5_kna':'ङ','character_1_ka':'क','character_31_petchiryakha':'ष','sp_char_27':"र"}

ind_2_name={0: 'character_10_yna',
 1: 'character_11_taamatar',
 2: 'character_12_thaa',
 3: 'character_13_daa',
 4: 'character_14_dhaa',
 5: 'character_15_adna',
 6: 'character_16_tabala',
 7: 'character_17_tha',
 8: 'character_18_da',
 9: 'character_19_dha',
 10: 'character_1_ka',
 11: 'character_20_na',
 12: 'character_21_pa',
 13: 'character_22_pha',
 14: 'character_23_ba',
 15: 'character_24_bha',
 16: 'character_25_ma',
 17: 'character_26_yaw',
 18: 'character_27_ra',
 19: 'character_28_la',
 20: 'character_29_waw',
 21: 'character_2_kha',
 22: 'character_30_motosaw',
 23: 'character_31_petchiryakha',
 24: 'character_32_patalosaw',
 25: 'character_33_ha',
 26: 'character_34_chhya',
 27: 'character_35_tra',
 28: 'character_36_gya',
 29: 'character_3_ga',
 30: 'character_4_gha',
 31: 'character_5_kna',
 32: 'character_6_cha',
 33: 'character_7_chha',
 34: 'character_8_ja',
 35: 'character_9_jha',
 36: 'sp_char_4',
 37: 'sp_char_5',
 38: 'sp_char_1',
 39: 'sp_char_12',
 40: 'sp_char_36',
 41: 'sp_char_40',
 42: 'sp_char_27',
 43: 'sp_char_37'  }

print('Model loaded. Check http://127.0.0.1:5000/')

app.secret_key = "secret key"
UPLOAD_FOLDER='static/uploads/'
def model_predict(img,model):
    preds=model.predict(img)
    return preds

@app.route('/')

def hello_world():
    return render_template('index.html')

@app.route('/')
def upload_form():
	return render_template('upload.html')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        # return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        # return redirect(request.url)
 
        # Make prediction
    # print(filename+r"%%%%%$")
    s=""
    reader = easyocr.Reader(['hi']) 
    bounds = reader.readtext('C:\\Users\\Rohith\\Desktop\\Akshara-SIH-main\\static\\uploads\\'+filename, detail=1) 
    chars=""
    flag=1
    print(bounds)
    for i in bounds:
        if(i[2]<0.75):
            flag=0
        chars+=i[1]
        chars+=" "

    li=segment(filename)
    # print(li)
    for i in li:
        # print(i.shape)
        if(i.shape[0]==0):
            continue
        preds = model_predict(i, model)
        for j in preds:
            s=s+di[ind_2_name[j.argmax()]]
            # print(s)
        s=s+' '
    values=s
    a=values
    if(flag==1 or len(chars.split(" "))>len(values.split(" "))):
        print(len(chars.split(" ")),len(values.split(" ")))
        values=chars
    return render_template("index.html", value=values)
    # return s

def segment(img):
    xl=[]
    x='C:\\Users\\Rohith\\Desktop\\Akshara-SIH-main\\static\\uploads\\'
    # print(x+char+img)
    img = cv2.imread(x+img)#change
    
    # img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape

    if w > 1000:
        
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    def thresholding(image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
        # plt.imshow(thresh, cmap='gray')
        return thresh
    thresh_img = thresholding(img)

    #dilation
    kernel = np.ones((3,85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
    # plt.imshow(dilated, cmap='gray')

    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

    img2 = img.copy()
    def l1(z):
        for ctr in sorted_contours_lines:
            
            x,y,w,h = cv2.boundingRect(ctr)
            if(h<10):
                continue
            cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 0)
            roi = img2[y:y+h,x:x+w]
            # break
            # cv2_imshow(roi)
            # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            z.append(roi)
    l=[]
    a=l1(l)
    def borders(here_img, thresh, bthresh=0.092):
        shape = here_img.shape
        #check = int(115 * size[0] / 600)
        #check = int(55 * size[0] / 600)
        check= int(bthresh*shape[0])
        image = here_img[:]
        top, bottom = 0, shape[0] - 1
        #find the background color for empty column
        bg = np.repeat(thresh, shape[1])
        count = 0
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).any()) == True:
                #print(count)
                count += 1
            else:
                count = 0
            if count >= check:
                top = row - check
                break
        bg = np.repeat(thresh, shape[1])
        count = 0
        rows = np.arange(1, shape[0])
        #print(rows)
        for row in rows[::-1]:
            if  (np.equal(bg, image[row]).any()) == True:
                count += 1
            else:
                count = 0
            if count >= check:
                bottom = row + count
                break

        d1 = (top - 2) >= 0 
        d2 = (bottom + 2) < shape[0]
        d = d1 and d2
        if(d):
            b = 2
        else:
            b = 0
        
        return (top, bottom, b)
    def preprocess(bgr_img):#gray image   
        blur = cv2.GaussianBlur(bgr_img,(5,5),0)
        ret,th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #converts black to white and inverse

        rows, cols = th_img.shape
        # if(min(rows,cols)<5):
        #   return 
        bg_test = np.array([th_img[i][i] for i in range(5)])
        if bg_test.all() == 0:
            text_color = 255
        else:
            text_color = 0
        
        tb = borders(th_img, text_color)
        lr = borders(th_img.T, text_color)
        dummy = int(np.average((tb[2], lr[2]))) + 2
        template = th_img[tb[0]+dummy:tb[1]-dummy, lr[0]+dummy:lr[1]-dummy]
        
        # plt.imshow(template)
        # plt.show()
        return (template, tb, lr)
    def segmentation(bordered, thresh=255, min_seg=2.22, scheck=0.015):
        try:
            shape = bordered.shape
            # print(shape)
            # print(shape)
            check = int(scheck * shape[0])
            image = bordered[:]
            image = image[check:].T
            shape = image.shape
            #plt.imshow(image)
            #plt.show()

            #find the background color for empty column
            bg = np.repeat(255 - thresh, shape[1])
            bg_keys = []
            for row in range(1, shape[0]):
            # cv2_imshow(image[row]-255)
            # print(image[row]-255)
            # print(bg)
                if  (np.equal(bg, image[row]).all()):
                        bg_keys.append(row)            

            lenkeys = len(bg_keys)-1
            if len(bg_keys)>1:
                new_keys = [bg_keys[1], bg_keys[-1]]
            else:
                new_keys=[0,shape[1]]
            #print(lenkeys)
            for i in range(1, lenkeys):
                if (bg_keys[i+1] - bg_keys[i]) > check:
                    new_keys.append(bg_keys[i])
                    #print(i)

            new_keys = sorted(new_keys)
            #print(new_keys)
            segmented_templates = []
            first = 0
            bounding_boxes = []
            for key in new_keys[1:]:
                segment = bordered.T[first:key]
                if segment.shape[0]>=min_seg and segment.shape[1]>=min_seg:
                    segmented_templates.append(segment.T)
                    if(first-key>0.8*shape[0]):
                        continue
                    bounding_boxes.append((first, key))
                first = key
            
            last_segment = bordered.T[new_keys[-1]:]
            if last_segment.shape[0]>=min_seg and last_segment.shape[1]>=min_seg:
                segmented_templates.append(last_segment.T)
                bounding_boxes.append((new_keys[-1], new_keys[-1]+last_segment.shape[0]))


            return(segmented_templates, bounding_boxes)
        except:
            # print(1)
            return [bordered, (0, bordered.shape[1])]
    def check(img):
        number_of_white_pix = np.sum(img == 255)
        number_of_black_pix = np.sum(img == 0)
        return number_of_white_pix/number_of_black_pix
    # img=cv2.imread('/content/sample_data/a1.jpeg',0)
    # cv2_imshow(img)
    # a=[]
    for i in l:
        img=i
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prepimg, tb, lr = preprocess(gray_image)
        # cv2_imshow(prepimg)
        segments=segmentation(prepimg,255)
        l1=[]
        char_images1=[]
        for simg in segments[0]:
            # print(simg)
            # print(simg.shape)
            # simg=cv2.resize(simg,(32,32))
            # cv2_imshow(simg)
            # plt.imshow(simg)
            # k=segmentation(simg)
            # prepimg, tb, lr = preprocess(simg)
            l1.append(simg)
            # for i in k[0]:
            #   print(i.shape)
            # cv2_imshow(i for i in l[0])

            # plt.show()
        for i in l1:
            k=segmentation(i,255,2.2222,0.3333)
            # print(k[0])
            for j in k[0]:
            # print(j.shape)
            # j=cv2.resize(j,(32,32))
            # j=j.astype('float32')/255
                char_images1.append(j)
            # cv2_imshow(j)
        a=[]
        for i in char_images1:
            char_images=[]
            
            k=segmentation(i,255,2,0.195)
            # print(k[0])
            for j in k[0]:
                # print(j.shape)
                j=cv2.resize(j,(32,32))
                # j=j.astype('float32')/255
                char_images.append(j)
                # print('hlo')
                # cv2_imshow(j)
            x=check(char_images[0])
            # print(x)
            # print(len(char_images))
            for q in range(len(char_images)):
                if(x>0.08 and x<0.9):
                    a.append(char_images[q])
        x_train=np.zeros((len(a),32,32),dtype=np.uint8)
        print("Lena=",len(a))
        for i in range(len(a)):
            x_train[i]=a[i]
        x_train=x_train.astype('float32')/255
        x_train=x_train.reshape((-1,32,32,1))
        xl.append(x_train)
    return xl
# main driver function
if __name__ == '__main__':
    app.run()