# import urllib
import urllib.request

import cv2
import numpy as np

from wide_resnet import WideResNet

# Replace the URL with your own IPwebcam shot.jpg IP:port
url = 'http://192.168.1.157:8080/shot.jpg'
face_cascade = cv2.CascadeClassifier(
    '/home/system/opencv/data/haarcascades/haarcascade_frontalface_default.xml')


def crop_face(imgarray, section, margin=40, size=64):

    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w-1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h-1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(
        cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


def initialize_caffe_models():
    model = WideResNet(64)()
    model.load_weights(
        '/home/system/imagezie_reduce/pretrained_models/weights.18-4.06.hdf5')
    return model


def processor(img):
    model = initialize_caffe_models()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_percent = 40  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # img = cv2.flip(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    # To draw a rectangle in a face
    #    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    #    face_img = img[y:y+h, h:h+w].copy()
    # face_img.resize(64, 64, 3)
    # face_img = np.expand_dims(face_img, 0)
    # esults = model.predict(face_img)
    # predicted_genders = results[0]
    # ages = np.arange(0, 101).reshape(101, 1)
    # predicted_ages = results[1].dot(ages).flatten()
    # label = "{}, {}".format(int(predicted_ages[0]),
    #                       "F" if predicted_genders[0][0] > 0.5 else "M")
    # print(label)
    # blob = cv2.dnn.blobFromImage(
    #     face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # # Predict Gender
    # gender_net.setInput(blob)
    # gender_preds = gender_net.forward()
    # gender = gender_list[gender_preds[0].argmax()]
    # print("Gender : " + gender)

    # # Predict Age
    # age_net.setInput(blob)
    # age_preds = age_net.forward()
    # age = age_list[age_preds[0].argmax()]
    # print("Age Range: " + age)

 #       overlay_text = label
 #       cv2.putText(img, overlay_text, (x, y), font,
  #                  1, (255, 255, 255), 2, cv2.LINE_AA)

    # face_imgs = np.empty(
    #     (len(faces), 64, 64, 3))
    face_imgs = []
    for (x, y, w, h) in faces:
        # face_img, cropped = crop_face(
         #   frame, face, margin=40, size=64)
        # (x, y, w, h) = cropped
        face_img = frame[y:y+h, h:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        print(face_img.shape[2])
        # face_img = face_img.reshape(64, 64, 3)
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', face_img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        face_imgs.append([face_img])
        if len(face_imgs) > 0:
            #     # predict ages and genders of the detected faces
            results = model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            # # draw results
            for i, face in enumerate(faces):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                overlay_text = label
                cv2.putText(frame, overlay_text, (face[0], face[1]), font,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            # # cv2.imshow('Keras Faces', frame)

    return frame


# age_net, gender_net = initialize_caffe_models()
while True:
        # Use urllib to get the image and convert into a cv2 usable format
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    #img = cv2.imread('kuwait2 (1).jpg')
    img = cv2.imdecode(imgNp, -1)
    img = processor(img)
    # img = cv2.imread('test.jpeg')
    # Display an image in a window
    cv2.imshow('img', img)
    # cv2.imwrite('test.jpeg', img)

    # cv2.imshow('img', gray)

    # To give the processor some less stress
    # time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
