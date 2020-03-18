import os,sys
import re
import json
import urllib.request
import zipfile
import hashlib
import subprocess
import uuid
import time
import zipfile

from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
os.environ["TF_KERAS"] = "1"

from vis.utils import utils
from vis.visualization import visualize_cam, overlay

import tensorflow as tf
import numpy as np
import skimage

from tensorflow.python import keras

from tensorflow.contrib.data import assert_element_shape
from tensorflow.python.keras import backend as K, Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import utils
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.callbacks import History, TensorBoard, LearningRateScheduler

from keras_radam import RAdam

import oss2

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from tools.utils import Helper, create_loss_fn, INFO, ERROR, NOTE
from tools.custom import Yolo_Precision, Yolo_Recall
from models.yolonet import *

auth = oss2.Auth('MEOWMEOWMEOW', 'MEOWMEOWMEOW')
bucket = oss2.Bucket(auth, 'MEOWMEOWMEOW', 'MEOWMEOWMEOW')

upstreamServerAddress = "MEOWMEOWMEOW"
upstreamMagic = "MEOWMEOWMEOW"

aliyunOSSAddress = 'MEOWMEOWMEOW'

localSSDLoc = ''
nncaseLoc = f'MEOWMEOWMEOW'

senderAddr = "MEOWMEOWMEOW"
senderPass = "MEOWMEOWMEOW"

bootFileName = "MEOWMEOWMEOW"

detectionBootFileName = "MEOWMEOWMEOW"
detectionBinFile = "MEOWMEOWMEOW"
flashListFileName = 'MEOWMEOWMEOW'

completedTaskHeader = ""

def loss_softmax_cross_entropy_with_logits_v2(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct,logits=predicted)

def getDataset(address, uuid):
    try:
        urllib.request.urlretrieve(address, f"{localSSDLoc}dataset/{uuid}_dataset.zip")
        md5 = hashlib.md5(open(f"{localSSDLoc}dataset/{uuid}_dataset.zip", 'rb').read()).hexdigest()
        with zipfile.ZipFile(f"{localSSDLoc}dataset/{uuid}_dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{localSSDLoc}dataset_tmp/{uuid}_dataset")

        return(0, f"{localSSDLoc}dataset_tmp/{uuid}_dataset", md5)
    except Exception as e:
        return(-1, f"Failed to fetch the dataset, dut to {e}", 'NaN')


def chkWaterMark(filename, magic):
    m = hashlib.sha256()
    m.update(str.encode(f"{magic} is super cool!"))

    try:
        fsize = os.path.getsize(filename)
        f = open(filename, 'rb')
        f.seek(fsize - 32)

        wtmData = f.read()
        f.close()
    except Exception as e:
        return(-1, f"Failed to fetch the dataset, dut to {e}")

    if wtmData == m.digest():
        return(0, wtmData)
    else:
        return(-7, "Sorry, We currently only serivce M5StickV Customers.")

def chkDataset(dirname):

    listDatasetDir = os.listdir(dirname)

    if "v-training.config" in listDatasetDir:
        print('Found config file, using detection mode.')
        try:
            vdetConfig = json.load(open(os.path.join(dirname, "v-training.config")))
            numOfClass = vdetConfig['classes']
            #key = vdetConfig['key']
        except Exception as e:
            return(-34, 'Failed to read config file, please make sure you have the correct format. Err:', e)

        numOfDataset = 0
        fileHash = []
        for fileName in listDatasetDir:
            if '.jpg' in fileName and (fileName.replace('.jpg', '.txt')) not in listDatasetDir:
                return(-35, f"Unmatched pair found in dataset. Found {fileName}, but {fileName.replace('.jpg', '.txt')} is not exist.")

            if '.jpg' in fileName:
                fileMD5 = hashlib.md5(open(os.path.join(dirname, fileName), 'rb').read()).hexdigest()
                if fileMD5 in fileHash:
                    return(-36, f"Repeated image {fileName} found in dataset. You should remove the repeated image for ensuring the quaility of model.")

                try:
                    im = Image.open(os.path.join(dirname, fileName))
                    #if(im.size != (320, 240)):
                    #    return(-2, f"Unexpected Image Size, only accpet 320x240 from M5Stick series products, but you have {im.size}")
                    im.close()
                except:
                    return(-37, f"Failed to open image {fileName}, plase make sure the images are not corrupted.")

                try:
                    labelFile = open(os.path.join(dirname, fileName.replace('.jpg', '.txt')), 'r')
                    for labelLine in labelFile:
                        labels = labelLine.split(' ')

                        if int(labels[0]) >= numOfClass:
                            return(-38, f"Label index overflowed, in {labelFile}, you claimed {numOfClass}, but it index {labels[0]}.")

                        if float(labels[1]) > 1.0 or float(labels[1]) < 0.0 or float(labels[2]) > 1.0 or float(labels[2]) < 0.0:
                            return(-39, f"Label bbox overflowed, the x,y,w,h should be in reletive mode and in the range of [0, 1].")

                    labelFile.close()
                except Exception as e:
                    return(-40, f"Failed to open label file {fileName.replace('.jpg', '.txt')}, plase make sure the label files are not corrupted. Err:", e)

                numOfDataset = numOfDataset + 1

        if numOfDataset < 30:
            return(-37, f"We require you upload more than 30 labled image for each training requres, but you only have {numOfDataset}")

        return (0, numOfClass, 'detection', vdetConfig)

    else:
        print('Config file not found, using classification mode.')

        if "train" not in listDatasetDir or ("vaild" not in listDatasetDir and "valid" not in listDatasetDir):
            return(-8, "Train or Valid folder not found. If you are using the M5StickV software, make sure you reach enough image counts of 35 per class.")

        if "valid" in listDatasetDir:
            os.rename(f"{dirname}/valid", f"{dirname}/vaild")

        TrainImageNum = 0
        VaildImageNum = 0

        NumOfClass = len(os.listdir(os.path.join(dirname, "train")))
        NumOfClassVaild = len(os.listdir(os.path.join(dirname, "vaild")))

        if NumOfClass != NumOfClassVaild:
            return(-11, "Number of Classes presented in Train and Vaild dataset is not equal.")

        print("[INFO]", f"Total {NumOfClass} Classes Found")

        if NumOfClass < 2:
            return(-16, "Number of Classes should larger or equal to two.")

        for file in listDatasetDir:
            if(os.path.splitext(file)[1] == '.dpf'):
                isConfigFilePresent = 1
                with open(os.path.join(dirname, file), 'r') as f:
                    datasetCfg = json.load(f)
        try:
            for dirpath, _ , filenames in os.walk(f"{dirname}/train", topdown=False):
                for file in filenames:
                    if(os.path.splitext(file)[1] == '.jpg'):
                        im=Image.open(os.path.join(dirpath, file))
                        #if(im.size != (320, 240)):
                        #    return(-2, f"Unexpected Image Size, only accpet 320x240, you have {im.size}")
                        im.close()
                        TrainImageNum = TrainImageNum + 1
                    #else:
                    #   return(-3, f"Unexpected File Present: {file}")

            for dirpath, _ , filenames in os.walk(f"{dirname}/vaild", topdown=False):
                for file in filenames:
                    if(os.path.splitext(file)[1] == '.jpg'):
                        im=Image.open(os.path.join(dirpath, file))
                        #if(im.size != (320, 240)):
                        #    return(-2, f"Unexpected Image Size, only accpet 320x240, you have {im.size}")
                        im.close()
                        VaildImageNum = VaildImageNum + 1
                    #else:
                    #   return(-3, f"Unexpected File Present: {file}")
        except Exception as e:
            return(-6, f"Unexpected error happened during checking dataset, {e}")

        if TrainImageNum < NumOfClass * 30:
            return(-4, f"Lake of Enough Train Dataset, Only {TrainImageNum} pictures found, but you need {NumOfClass * 30} in total.")

        if VaildImageNum < NumOfClass * 5:
            return(-4, f"Lake of Enough Valid Dataset, Only {VaildImageNum} pictures found, but you need {NumOfClass * 5} in total.")

        #if isConfigFilePresent == 0: #Not enable isConfigFilePresent Yet
        #    return(-5, "Unable to find dpf dataset description file.")

        return (0, NumOfClass, 'classification')

def runTrainingDetection(uuid, datasetDir, numOfClass, obj_thresh = 0.7, iou_thresh = 0.5, obj_weight = 1.0, noobj_weight = 1.0, wh_weight = 1.0, max_nrof_epochs = 50, batch_size = 96, vaildation_split = 0.2):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    datasetList = [os.path.join(datasetDir, f) for f in os.listdir(datasetDir)]
    image_list = []

    #img_ann_f = open(os.path.join(datasetDir, 'dataset_img_ann.txt'), 'w+')

    for fileName in datasetList:
        if '.jpg' in fileName:
    #        print('/home/m5stack/VTrainingService/' + fileName, file=img_ann_f)
            image_list.append('/home/m5stack/VTrainingService/' + fileName)

    #img_ann_f.close()

    image_path_list = np.array(image_list)#np.loadtxt(os.path.join(datasetDir, 'dataset_img_ann.txt'), dtype=str)

    ann_list = list(image_path_list)
    ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
    ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]

    lines = np.array([
        np.array([
            image_path_list[i],
            np.loadtxt(ann_list[i], dtype=float, ndmin=2),
            np.array(skimage.io.imread(image_path_list[i]).shape[0:2])]
        ) for i in range(len(ann_list))])

    np.save(os.path.join(datasetDir, 'dataset_img_ann.npy'), lines)

    #print('dataset npu>>>', os.path.join(datasetDir, 'dataset_img_ann.npy'))

    h = Helper(os.path.join(datasetDir, 'dataset_img_ann.npy'), numOfClass, 'voc_anchor.npy',
        np.reshape(np.array((224, 320)), (-1, 2)), np.reshape(np.array((7, 10, 14, 20)), (-1, 2)), vaildation_split)

    h.set_dataset(batch_size, 6)

    network = eval('yolo_mobilev1')  # type :yolo_mobilev2
    yolo_model, train_model = network([224, 320, 3], len(h.anchors[0]), numOfClass, alpha=0.5)

    train_model.compile(
        RAdam(),
        loss=[create_loss_fn(h, obj_thresh, iou_thresh, obj_weight, noobj_weight, wh_weight, layer)
            for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)],
        metrics=[Yolo_Precision(obj_thresh, name='p'), Yolo_Recall(obj_thresh, name='r')])

    shapes = (train_model.input.shape, tuple(h.output_shapes))
    h.train_dataset = h.train_dataset.apply(assert_element_shape(shapes))
    h.test_dataset = h.test_dataset.apply(assert_element_shape(shapes))

    #print('train', h.train_dataset, '\n\r\n\rtest', h.test_dataset)

    try:
        train_model.fit(h.train_dataset, epochs=max_nrof_epochs,
                    steps_per_epoch=10,
                    validation_data=h.test_dataset, validation_steps=1)
    except Exception as e:
        return(-45, 'Unexpected error found during training, err:', e)

    keras.models.save_model(yolo_model, f'{localSSDLoc}trained_h5_file/{uuid}_mbnet5_yolov3.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model_file(f'{localSSDLoc}trained_h5_file/{uuid}_mbnet5_yolov3.h5', custom_objects={'RAdam': RAdam, 'loss_softmax_cross_entropy_with_logits_v2':loss_softmax_cross_entropy_with_logits_v2})
    tflite_model = converter.convert()
    open(f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet5_yolov3_quant.tflite', "wb").write(tflite_model)

    subprocess.run([f'{nncaseLoc}/ncc', f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet5_yolov3_quant.tflite', f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet5_yolov3.kmodel', '-i', 'tflite', '-o', 'k210model', '--dataset', datasetDir])

    if os.path.isfile(f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet5_yolov3.kmodel'):
        return (0, f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet5_yolov3.kmodel')
    else:
        return (-16, 'Unexpected Error Found During generating Kendryte k210model.')



def runTrainingClassification(uuid, datasetDir, validDir, classNum, dropoutValue = 0.2, batch_size = 128, nb_epoch = 20, step_size_train = 10, alphaVal = 0.75, depthMul = 1):

    imageGen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.35,
        height_shift_range=0.35,
        zoom_range=0.35,
        shear_range=0.35,
        vertical_flip=False,
        horizontal_flip=False,
        brightness_range = [0.65, 1.35],
        rescale=1. / 255)

    trainSet=imageGen.flow_from_directory(datasetDir,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)
    validSet=imageGen.flow_from_directory(validDir,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical', shuffle=True)

    class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
        def __init__(self, patience=3):
            super(EarlyStoppingAtMinLoss, self).__init__()
            self.patience = patience
            self.best_weights = None

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0
            self.best = np.Inf
            self.last_acc = 0
            self.atleastepoc = 0

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get('val_loss')
            val_acc = logs.get('val_acc')
            self.atleastepoc = self.atleastepoc + 1
            if np.less(current, self.best) or self.last_acc < 0.95 or self.atleastepoc < 25:
                self.best = current
                self.wait = 0
                self.last_acc = val_acc
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print('\nRestoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    base_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), alpha = alphaVal,depth_multiplier = depthMul, dropout = dropoutValue, pooling='avg',include_top = False, weights = "imagenet", classes = classNum)

    mbnetModel = Sequential([
        base_model,
        Dropout(dropoutValue, name='dropout'),
        Dense(classNum, activation='softmax')
    ])


    if classNum == 2:
        mbnetModel.compile(loss='binary_crossentropy',
                optimizer=RAdam(),
                metrics=['accuracy'])
    else:
        mbnetModel.compile(loss='categorical_crossentropy',#loss_softmax_cross_entropy_with_logits_v2,
                optimizer=RAdam(),
                metrics=['accuracy'])

    history = History()

    try:
        mbnetModel.fit_generator(generator=trainSet,steps_per_epoch=step_size_train,callbacks=[EarlyStoppingAtMinLoss(), history],epochs=50, validation_data=validSet)
    except Exception as e:
        return(-14, f'Unexpected Error Found During Triaining, {e}')

    mbnetModel.save(f'{localSSDLoc}trained_h5_file/{uuid}_mbnet10.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model_file(f'{localSSDLoc}trained_h5_file/{uuid}_mbnet10.h5', custom_objects={'RAdam': RAdam, 'loss_softmax_cross_entropy_with_logits_v2':loss_softmax_cross_entropy_with_logits_v2})
    tflite_model = converter.convert()
    open(f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet10_quant.tflite', "wb").write(tflite_model)

    subprocess.run([f'{nncaseLoc}/ncc', f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet10_quant.tflite', f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel', '-i', 'tflite', '-o', 'k210model', '--dataset', validDir])

    if os.path.isfile(f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel'):
        return (0, f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel', history, validSet, mbnetModel)
    else:
        return (-16, 'Unexpected Error Found During generating Kendryte k210model.')

def packImages(uuid, history, validSet, mbnetModel):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.ylim(0, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.autoscale()
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(f'{localSSDLoc}imageAssest/{uuid}_acc_graph.png')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.autoscale()
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'{localSSDLoc}imageAssest/{uuid}_loss_graph.png')

def uploadFile(filename):
    try:
        with open(filename, 'rb') as fileobj:
            bucket.put_object(os.path.basename(filename), fileobj)
    except:
        return(-13, "Fail to upload file to server.")
    return (0, f'{aliyunOSSAddress}/{os.path.basename(filename)}')

def sendErrorEmail(mailaddr, userid, content):
    msg_content = 'Hi!\n\rSorry for that, there is some error happened during the training process. '\
        '\nWe attached the reason below, if you have any questions, welcome to report to us through Twitter, Facebook, or Forum.\n'\
        'Or, check out our docs here: https://docs.m5stack.com/#/en/related_documents/v-training'\
        f'\n\rUSERID: {userid}\nCONTENT: {content}'\

    message = MIMEText(msg_content, 'plain')

    message['From'] = f'V-Trainer <{senderAddr}>'
    message['To'] = f'{mailaddr} <{mailaddr}>'
    message['Subject'] = f'[V-Trainer] {userid} Online Training Request Failed'

    msg_full = message.as_string()

    server = smtplib.SMTP_SSL()
    #server.set_debuglevel(1)
    server.connect(host='smtp.qiye.aliyun.com',port=465)
    server.login(senderAddr, senderPass)
    server.sendmail(senderAddr,
                    [mailaddr],
                    msg_full)
    server.quit()

def sendSuccessEmail(mailaddr, userid, ossAddr, typeOfRunning):
    msgRoot = MIMEMultipart('related')
    msgRoot['From'] = f'V-Trainer <{senderAddr}>'
    msgRoot['To'] = f'{mailaddr} <{mailaddr}>'
    msgRoot['Subject'] = f'[V-Trainer] {userid} Online Training Request Finished'

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)

    if typeOfRunning == 'classification':
        msg_content = f'Hi!\n\nYour training request have been successfully processed, you can download the kmodel & sample program files here: \n{ossAddr}\n\r\n\r'\
                    "Model: Classification MobileNetV1 Alpha: 0.7 Depth: 1"

        msgText = MIMEText(msg_content, 'plain')
        msgAlternative.attach(msgText)
    # '''
        msgText = MIMEText(msg_content.replace("\n", "<br>") + '<br><img src="cid:acc_graph"><br><img src="cid:loss_graph"><br><img src="cid:conf_matrix"><br>', 'html')
        msgAlternative.attach(msgText)

        with open(f'{localSSDLoc}imageAssest/{userid}_loss_graph.png', 'rb') as imageFile:
            msgImage = MIMEImage(imageFile.read())
            msgImage.add_header('Content-ID', '<loss_graph>')
            msgRoot.attach(msgImage)

        with open(f'{localSSDLoc}imageAssest/{userid}_acc_graph.png', 'rb') as imageFile:
            msgImage = MIMEImage(imageFile.read())
            msgImage.add_header('Content-ID', '<acc_graph>')
            msgRoot.attach(msgImage)
    else:
        msg_content = f'Hi!\n\nYour training request have been successfully processed, you can download the kmodel & sample program files here: \n{ossAddr}\n\r\n\r'\
                        "Model: Detection YoloV3 Backbone: MobileNetV1 Alpha: 0.5 Depth: 1\n\r\n\r"\
                        "In order to run the model without crash, you need to download the attached flash to your M5 devices.\n\r"\
                        "(To be awared, although every time the attached maixpy.bin should be the same, we may update it without notificaiton.)\n\r\n\r"\
                        "Have fun with the new v-training detectio :p\n\r"\
                        "if you have any questions, welcome to report to us through Twitter, Facebook, or Forum.\n\r"

        msgText = MIMEText(msg_content, 'plain')
        msgAlternative.attach(msgText)

    server = smtplib.SMTP_SSL()
    server.set_debuglevel(1)
    server.connect(host='smtp.qiye.aliyun.com',port=465)
    server.login(senderAddr, senderPass)
    server.sendmail(senderAddr,
                    [mailaddr],
                    msgRoot.as_string())
    server.quit()

bootFileContent = open(bootFileName, "r").read()
blackListEmail = []
fileMD5Hash = {}

while 1:
    try:

    #print("===================REV REQUEST=====================")

        try:
            if completedTaskHeader != "":
                print(">>> Req With TaskHeader: " + completedTaskHeader)
            req = urllib.request.Request(upstreamServerAddress)
            req.add_header('Basic', upstreamMagic)
            req.add_header('X-Complete-Task', completedTaskHeader)
            r = urllib.request.urlopen(req).read()

            completedTaskHeader = ""

            jsonRequest = json.loads(r.decode('utf-8'))

            userid = jsonRequest['id']

        except:
            print("[FATAL]", "Failed to fetch request from server")
            continue

        if jsonRequest['id'] == "":
            #print("[INFO]", "No pending works...")
            time.sleep(1)
        else:
            print("===================REV REQUEST=====================")
            print("[INFO]", "Get work! Json String:", r)

            if jsonRequest['email'] in blackListEmail:
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], 'Service not available.')
                completedTaskHeader = str(userid)
                continue

            print("[INFO]", "Start Downloading Dataset, Address: ", jsonRequest['url'])
            ret = getDataset(jsonRequest['url'], jsonRequest['id'])
            datasetDir = ret[1]

            md5 = ret[2]

            if md5 in fileMD5Hash and ret[0] == 0:
                fileMD5Hash[md5] = fileMD5Hash[md5] + 1
                if fileMD5Hash[md5] == 3:
                    sendErrorEmail(jsonRequest['email'], jsonRequest['id'], 'Dataset has been uploaded too many times.')
                    completedTaskHeader = str(userid)
                    continue
            else:
                fileMD5Hash[md5] = 0

            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                completedTaskHeader = str(userid)
                continue

            print("[INFO]", "Start Checking Dataset, dirname: ", ret[1])
            ret = chkDataset(ret[1])
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                completedTaskHeader = str(userid)
                continue

            numOfClass = ret[1]
            typeOfRunning = ret[2]

            print("[INFO]", f"Start {typeOfRunning} Training")
            if typeOfRunning == 'classification':
                ret = runTrainingClassification(jsonRequest['id'], f"{datasetDir}/train", f"{datasetDir}/vaild", numOfClass, 1/numOfClass)
                if ret[0] != 0:
                    print("[FATAL]", ret[1])
                    sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                    completedTaskHeader = str(userid)
                    continue

                try:
                    packImages(jsonRequest['id'], ret[2], ret[3], ret[4])
                except Exception as e:
                    print("[FATAL]", "Unable to generate Images," + str(e))
                    sendErrorEmail(jsonRequest['email'], jsonRequest['id'], "Unable to generate Images," + str(e))
                    completedTaskHeader = str(userid)
                    continue

                customBootFile = open(bootFileName, 'r').read()

            else:
                vdetConfig = ret[3]

                vdetDict = ['obj_thresh', 'iou_thresh', 'obj_weight', 'noobj_weight', 'wh_weight', 'max_nrof_epochs', 'batch_size', 'vaildation_split']
                vdetDebugDict = {}
                for vdetKey in vdetDict:
                    if vdetKey in vdetConfig:
                        print(f'[INFO] Found override parameter {vdetKey} = {vdetConfig[vdetKey]}')
                        vdetDebugDict[vdetKey] = vdetConfig[vdetKey]

                ret = runTrainingDetection(jsonRequest['id'], datasetDir, numOfClass, **vdetDebugDict)
                if ret[0] != 0:
                    print("[FATAL]", ret[1])
                    sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                    completedTaskHeader = str(userid)
                    continue

                customBootFile = open(detectionBootFileName, 'r').read()


            print("[INFO]", "Start Creating Custom Bootfile.")
            customBootFile = customBootFile.replace("{MODEL_NAME}", f"{os.path.basename(ret[1])}")
            labelsStr = ""

            for i in range(1, numOfClass):
                labelsStr = labelsStr + f"\"{i}\","
            labelsStr = labelsStr + f"\"{numOfClass}\""

            customBootFile = customBootFile.replace("{MODEL_LABELS}", labelsStr)

            os.mkdir(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}")
            open(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py", "w+").write(customBootFile)

            if typeOfRunning == 'detection':
                customFlashListFile = open(flashListFileName, 'r').read()
                customFlashListFile = customFlashListFile.replace('{MODEL_NAME}', os.path.basename(ret[1]))
                open(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/flash-list.json", "w+").write(customFlashListFile)

            userid = jsonRequest['id']

            vtrainer_output = f'{localSSDLoc}output_file/{str(uuid.uuid4())}_{userid}_vtrainer.zip'

            print("[INFO]", f"Start Compressing All Files, \nzipdir: {vtrainer_output}\n"\
                f"bootfile Dir: {localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py"\
                f"kmodel Dir: {ret[1]}")

            with zipfile.ZipFile(vtrainer_output, 'w') as myzip:
                myzip.write(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py", "boot.py")

                if typeOfRunning == 'detection':
                    vtrainer_kfpkg = f'{localSSDLoc}output_file/{str(uuid.uuid4())}_{userid}_vtrainer.kfpkg'
                    with zipfile.ZipFile(vtrainer_kfpkg, 'w') as mykfpkg:
                        mykfpkg.write(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/flash-list.json", "flash-list.json")
                        mykfpkg.write(ret[1], os.path.basename(ret[1]))
                        mykfpkg.write(detectionBinFile, "maixpy_with_yolov3_patch.bin")

                    myzip.write(vtrainer_kfpkg, f"{userid}_vtrainer.kfpkg")
                else:
                    myzip.write(f"{localSSDLoc}startup.jpg", "startup.jpg")
                    myzip.write(ret[1], os.path.basename(ret[1]))

            print("[INFO]", "Start Uploading File, filename:", ret[1])
            ret = uploadFile(vtrainer_output)
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                completedTaskHeader = str(userid)
                continue

            print("[INFO]", "Start Sending Success Email, address:", ret[1])
            sendSuccessEmail(jsonRequest['email'], jsonRequest['id'], ret[1], typeOfRunning)
            completedTaskHeader = str(userid)


    except Exception as e:
        completedTaskHeader = str(userid)
        print("[FATAL] Unexpcted error happened during training. Err:", e)
        try:
            sendErrorEmail(jsonRequest['email'], jsonRequest['id'], 'Unexpcted error happened during training. Err: ' + str(e))
        except:
            pass
