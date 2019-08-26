import os
import json
import urllib.request
import zipfile
import hashlib
import subprocess
import uuid
import time
import zipfile

from PIL import Image

os.environ["TF_KERAS"] = "1"


import tensorflow as tf
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers import ZeroPadding2D
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

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.callbacks import History

from keras_radam import RAdam

import oss2

import smtplib
from email.mime.text import MIMEText

#auth = oss2.Auth('Meowmeowmeow', 'Meowmeowmeow')
auth = oss2.Auth('Meowmeowmeow', 'Meowmeowmeow')
bucket = oss2.Bucket(auth, 'Meowmeowmeow', 'Meowmeowmeow')

upstreamServerAddress = "Meowmeowmeow"
upstreamMagic = "Meowmeowmeow"

#aliyunOSSAddress = 'Meowmeowmeow'
aliyunOSSAddress = 'Meowmeowmeow'

localSSDLoc = 'Meowmeowmeow'
nncaseLoc = 'Meowmeowmeow'

senderAddr = "Meowmeowmeow"
senderPass = "Meowmeowmeow"

bootFileName = "Meowmeowmeow"

def getDataset(address, uuid):
    try:
        urllib.request.urlretrieve(address, f"{localSSDLoc}dataset/{uuid}_dataset.zip")

        with zipfile.ZipFile(f"{localSSDLoc}dataset/{uuid}_dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{localSSDLoc}dataset_tmp/{uuid}_dataset")
        
        return(0, f"{localSSDLoc}dataset_tmp/{uuid}_dataset")
    except Exception as e:
        return(-1, f"Failed to fetch the dataset, dut to {e}")

def chkDataset(dirname):  

    listDatasetDir = os.listdir(dirname)

    if "train" not in listDatasetDir or ("vaild" not in listDatasetDir and "valid" not in listDatasetDir):
        return(-8, "Train or valid folder not found. If you are using the M5StickV software, make sure you reach enough image counts of 35.")

    if "valid" in listDatasetDir:
        os.rename(f"{dirname}/valid", f"{dirname}/vaild")

    TrainImageNum = 0
    VaildImageNum = 0
    
    NumOfClass = len(os.listdir(os.path.join(dirname, "train")))
    NumOfClassVaild = len(os.listdir(os.path.join(dirname, "vaild")))
    
    if NumOfClass != NumOfClassVaild:
        return(-11, "Number of Classes presented in Train and Vaild dataset is not equal.")
    
    print("[INFO]", f"Total {NumOfClass} Classes Found")

    if NumOfClass < 3:
        return(-16, "Number of Classes should larger or equal to three.")

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
                else:
                    return(-3, f"Unexpected File Present: {file}")
        
        for dirpath, _ , filenames in os.walk(f"{dirname}/vaild", topdown=False):
            for file in filenames:
                if(os.path.splitext(file)[1] == '.jpg'):
                    im=Image.open(os.path.join(dirpath, file))
                    #if(im.size != (320, 240)):
                    #    return(-2, f"Unexpected Image Size, only accpet 320x240, you have {im.size}")
                    im.close()
                    VaildImageNum = VaildImageNum + 1
                else:
                    return(-3, f"Unexpected File Present: {file}")
    except Exception as e:
        return(-6, f"Unexpected error happened during checking dataset, {e}")

    if TrainImageNum < NumOfClass * 10:
        return(-4, f"Lake of Enough Dataset, Only {TrainImageNum} pictures found, but you need {NumOfClass * 50} in total.")
    
    if VaildImageNum < NumOfClass * 5:
        return(-4, f"Lake of Enough Dataset, Only {VaildImageNum} pictures found, but you need {VaildImageNum * 10} in total.")
    
    #if isConfigFilePresent == 0: #Not enable isConfigFilePresent Yet
    #    return(-5, "Unable to find dpf dataset description file.")
    
    return (0, NumOfClass)

def runTraining(uuid, datasetDir, validDir, classNum, dropoutValue = 0.2, batch_size = 64, nb_epoch = 20, step_size_train = 10, alphaVal = 1.0, depthMul = 1):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    tf.Session(config=config)

    imageGen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        shear_range=0.3,
        vertical_flip=False,
        horizontal_flip=False,
        rescale=1. / 255)
    
    trainSet=imageGen.flow_from_directory(datasetDir,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)
    validSet=imageGen.flow_from_directory(validDir,  
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)

    class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
        def __init__(self, patience=0):
            super(EarlyStoppingAtMinLoss, self).__init__()
            self.patience = patience
            self.best_weights = None

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0
            self.best = np.Inf 
            self.last_acc = 0

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get('val_loss')
            val_acc = logs.get('val_acc')
            if np.less(current, self.best) or self.last_acc < 0.8:
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

    base_model = MobileNet(input_shape=(224, 224, 3), alpha = alphaVal,depth_multiplier = depthMul, dropout = dropoutValue, pooling='avg',include_top = False, weights = "imagenet", classes = classNum)

    x = base_model.output
    x = Dropout(dropoutValue, name='dropout')(x)  
    output_layer = Dense(classNum, activation='softmax')(x)
    mbnetModel=Model(inputs=base_model.input,outputs=output_layer)
        
    mbnetModel.compile(loss='categorical_crossentropy',
                optimizer=RAdam(),
                metrics=['accuracy'])
    history = History()
    try:
        mbnetModel.fit_generator(generator=trainSet,steps_per_epoch=step_size_train,callbacks=[EarlyStoppingAtMinLoss(), history],epochs=nb_epoch, validation_data=validSet)
    except Exception as e:
        return(-14, f'Unexpected Error Found During Training, {e}')

    mbnetModel.save(f'{localSSDLoc}trained_h5_file/{uuid}_mbnet10.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model_file(f'{localSSDLoc}trained_h5_file/{uuid}_mbnet10.h5', custom_objects={'RAdam': RAdam})
    tflite_model = converter.convert()
    open(f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet10_quant.tflite', "wb").write(tflite_model)

    subprocess.run([f'{nncaseLoc}/ncc', f'{localSSDLoc}trained_tflite_file/{uuid}_mbnet10_quant.tflite', f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel', '-i', 'tflite', '-o', 'k210model', '--dataset', validDir])

    if os.path.isfile(f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel'):
        return (0, f'{localSSDLoc}trained_kmodel_file/{uuid}_mbnet10_quant.kmodel', history)
    else:
        return (-16, 'Unexpected Error Found During generating Kendryte k210model.')

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
    message['To'] = f'{userid} <{mailaddr}>'
    message['Subject'] = f'[V-Trainer] {userid} Online Training Request Failed'

    msg_full = message.as_string()

    server = smtplib.SMTP_SSL()
    server.set_debuglevel(1)
    server.connect(host='smtp.qiye.aliyun.com',port=465)  
    server.login(senderAddr, senderPass)
    server.sendmail(senderAddr,
                    [mailaddr],
                    msg_full)
    server.quit()

def sendSuccessEmail(mailaddr, userid, ossAddr, history):
    msg_content = f'Hi!\n\rYour training request have been successfully processed, you can download the kmodel & sample program files here: \r{ossAddr}'\
            f"\n\rYour Model Parameters:\nFinal Loss: {(history.history['loss'])[-1]}\nFinal Accurancy: {(history.history['acc'])[-1]}"\
            f"\nFinal Validation Loss: {(history.history['val_loss'])[-1]}\nFinal Validation Accurancy: {(history.history['val_acc'])[-1]}\nModel: Mobilenet V1 Alpha: 1.0 Depth: 1"
    if (history.history['val_acc'])[-1] < 0.75:
        msg_content = msg_content + "\n Your validation Accurancy seems too low, this may caused by bad dataset(the images you providing do not have enough similarity or just too less images)"\
    " or the optimzer do not converge. Under both circumstance, you can just put more images inside, and take another try!"
    message = MIMEText(msg_content, 'plain')

    message['From'] = f'V-Trainer <{senderAddr}>'
    message['To'] = f'{userid} <{mailaddr}>'
    message['Subject'] = f'[V-Trainer] {userid} Online Training Request Finished'

    msg_full = message.as_string()

    server = smtplib.SMTP_SSL()
    server.set_debuglevel(1)
    server.connect(host='smtp.qiye.aliyun.com',port=465)  
    server.login(senderAddr, senderPass)
    server.sendmail(senderAddr,
                    [mailaddr],
                    msg_full)
    server.quit()

bootFileContent = open(bootFileName, "r").read()

while 1:
    try:
        '''
        {
            "id":"714b3ae8cca31727",
            "url":"http://m5stickv-model.oss-cn-shenzhen.aliyuncs.com/2019-8-22/7cf4ae24aa71bbeee55ef9fc872b6c3f.zip",
            "datetime":1566473004992,
            "email":"a@b.c"
        }
        '''
        #print("===================REV REQUEST=====================")

        try:
            req = urllib.request.Request(upstreamServerAddress)
            req.add_header('Basic', upstreamMagic)
            r = urllib.request.urlopen(req).read()

            jsonRequest = json.loads(r.decode('utf-8'))
        except:
            print("[FATAL]", "Failed to fetch request from server")
            continue

        if jsonRequest['id'] == "":
            #print("[INFO]", "No pending works...")
            time.sleep(1)
        else:
            print("===================REV REQUEST=====================")
            print("[INFO]", "Get work! Json String:", r)

            print("[INFO]", "Start Downloading Dataset, Address: ", jsonRequest['url'])
            ret = getDataset(jsonRequest['url'], jsonRequest['id'])
            datasetDir = ret[1]
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                continue
            
            print("[INFO]", "Start Checking Dataset, dirname: ", ret[1])
            ret = chkDataset(ret[1])
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                continue
            
            numOfClass = ret[1]

            print("[INFO]", "Start Runiing Training")
            ret = runTraining(jsonRequest['id'], f"{datasetDir}/train", f"{datasetDir}/vaild", numOfClass, 1/numOfClass)
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                continue
            
            history = ret[2]

            print("[INFO]", "Start Creating Custom Bootfile.")
            customBootFile = bootFileContent
            customBootFile = customBootFile.replace("{MODEL_NAME}", f"{os.path.basename(ret[1])}")
            labelsStr = ""

            for i in range(1, numOfClass):
                labelsStr = labelsStr + f"\"{i}\","
            labelsStr = labelsStr + f"\"{numOfClass}\"" 

            customBootFile = customBootFile.replace("{MODEL_LABELS}", labelsStr)

            os.mkdir(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}")
            open(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py", "w+").write(customBootFile)
            
            userid = jsonRequest['id']

            vtrainer_output = f'{localSSDLoc}output_file/{str(uuid.uuid4())}_{userid}_vtrainer.zip'

            print("[INFO]", f"Start Compressing All Files, \nzipdir: {vtrainer_output}\n"\
                f"bootfile Dir: {localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py"\
                f"kmodel Dir: {ret[1]}")

            with zipfile.ZipFile(vtrainer_output, 'w') as myzip:
                myzip.write(f"{localSSDLoc}startup.jpg", "startup.jpg")
                myzip.write(ret[1], os.path.basename(ret[1]))
                myzip.write(f"{localSSDLoc}custom_bootfile/{jsonRequest['id']}/boot.py", "boot.py")

            print("[INFO]", "Start Uploading File, filename:", ret[1])
            ret = uploadFile(vtrainer_output)
            if ret[0] != 0:
                print("[FATAL]", ret[1])
                sendErrorEmail(jsonRequest['email'], jsonRequest['id'], ret[1])
                continue
            
            print("[INFO]", "Start Sending Success Email, address:", ret[1])
            sendSuccessEmail(jsonRequest['email'], jsonRequest['id'], ret[1], history)

    except Exception as e:
        print(e)
        time.sleep(1)
        pass
