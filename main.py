import cv2
import argparse
import configparser
import os
cwd=os.getcwd()
import torch
import torchvision
from PIL import Image
import json
import onnx
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt


def load_resnet_50(model_name):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    inference_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    saved_model='./'+model_name+'.pt'
    torch.save(inference_model, saved_model)
    return inference_model


def load_model(model_name):
# Create ResNet model (example)
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    inference_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    saved_model='./'+model_name+'.pt'
    torch.save(inference_model, saved_model)
    inference_model = pytorch_to_onnx((1,3,256,256), torch.float32, resnet50_model, model_name)
    gen_profile(resnet50_model_onnx, model_name)
    compile_dir = "resnet50_prepared"
    convert_to_fp16 = False
    use_int8_quantization_precision = True
    compile_qualcomm(inference_model, compile_dir, 4, 1, 1, 1, convert_to_fp16, use_int8_quantization_precision)
    validate_qualcomm(compile_dir)
    return inference_model


def center_crop(img):
    center = img.shape
    #print("center:", center)
    #print("height:", img.shape[1])
    #print("width:", img.shape[2])

    h = img.shape[1]
    w = img.shape[2]
    x = center[2]/2 - w/2
    y = center[1]/2 - h/2

    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    return crop_img

def image_to_tensor(image, dims, nchw):
#    img = cv2.imread(imagepath, cv2.IMREAD_COLOR)  #bgr

    #imdata = center_crop(img)
    #print("Shape after crop center:", img.shape)

    img = cv2.resize(np.array(image), dims,
                   interpolation=cv2.INTER_LINEAR).astype(np.float32)  #bgr
    imdata = np.asarray(img, dtype="float32")   #bgr

    if (nchw == True):
        # NHWC to NCHW
        imdata = imdata[:,:,0:3].transpose(2, 0, 1)   #bgr
        # Normalize
        imdata = np.asarray([(imdata[0,:,:]-102.255)*0.017429194,
                           (imdata[1,:,:]-116.28)*0.017507003,
                           (imdata[2,:,:]-123.675)*0.017124754],dtype="float32")
    else:
        imdata = np.asarray([(imdata[:,:,0]-103.94),
                           (imdata[:,:,1]-116.78),
                           (imdata[:,:,2]-123.68)],dtype="float32")
        imdata = np.reshape(imdata, (imdata.shape[1],imdata.shape[2],imdata.shape[0]))

    #print("Shape after resize:", imdata.shape)
    imdata = np.asarray([imdata], dtype="float32")
    #print("Shape after to numpy array:", imdata.shape)

    return imdata






def pytorch_to_onnx(shape, data_type, model, model_name):
    input = torch.randn(shape, dtype=data_type)
    try:
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        path_io = os.path.join(cwd, model_name + ".onnx")
        torch.onnx.export(model, input, path_io, opset_version=11)

        onnx_model = onnx.load(path_io)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as ' + model_name + '.onnx')

        return model_name + '.onnx'

    except Exception as e:
        print('ONNX export failure: %s' % e)




def gen_profile(model, model_name):
    try:
        cmd_str="/opt/qti-aic/exec/qaic-exec -m="+model+"  -dump-profile="+model_name+"-profile.yaml"
        subprocess.run(cmd_str,shell=True)
    except Exception as e:
        print('Failed to generate profile: %s' % e)

# Compile for Qualcomm Cloud AI100
def compile_qualcomm(model, compile_dir, num_cores, ols, mos, bs, fp16, int8):
    try:
        cmd_str="rm -rf "+ compile_dir
        subprocess.run(cmd_str,shell=True)
        number_cores=str(num_cores)
        ols_str=str(ols)
        bs_str=str(bs)
        mos_str=str(mos)
        if fp16:
            cmd_str="/opt/qti-aic/exec/qaic-exec -m="+model+" -aic-hw -aic-num-cores="+number_cores+" -ols="+ols_str+" -mos="+mos_str+" -batchsize="+bs_str+"  -convert-to-fp16   -aic-binary-dir="+compile_dir+ "-v   -compile-only"
            subprocess.run(cmd_str,shell=True)
            print('Model converted to FP16')
        elif int8:
            cmd_str="/opt/qti-aic/exec/qaic-exec -m="+model+" -aic-hw  -aic-num-cores="+number_cores+" -ols="+ols_str+" -mos="+mos_str+" -batchsize="+bs_str+" -quantization-precision='Int8'   -aic-binary-dir="+compile_dir+" -v   -compile-only"
            subprocess.run(cmd_str,shell=True)
            print('Model using Int8 Quantization')
        else:
            cmd_str="/opt/qti-aic/exec/qaic-exec -m="+model+" -aic-hw  -aic-num-cores="+number_cores+" -ols="+ols_str+" -mos="+mos_str+" -batchsize="+bs_str+" -aic-binary-dir="+compile_dir+" -v   -compile-only "
            subprocess.run(cmd_str,shell=True)
    except Exception as e:
        print('Model failed to compile: %s' % e)

# Model validation for Qualcomm hardware
def validate_qualcomm(model_bin):
    try:
        cmd_str="/opt/qti-aic/tools/qaic-qpc validate -i "+model_bin+"/programqpc.bin "
        subprocess.run(cmd_str,shell=True)
    except Exception as e:
        print('Failed to open model binary: %s' % e)

# Inference function for Qualcomm hardware
def predict_qualcomm(image, model_bin, output_dir, aic):
    try:
    #    from IPython.display import Image as Image2
     #   Image2(image)

        print("Converting image  to raw files...")
        #tensor = image_to_tensor(image, (224, 224), True) #original shape
        tensor = image_to_tensor(image, (256, 256), True) #updated shape for better accuracy
        #print(tensor)
        tensor.tofile('data/data.raw')

        #Try nvidia image conversion
        #img = Image.open(image)
        #preprocess = rn50_preprocess()
        #input_tensor = preprocess(img).unsqueeze(0) # create a mini-batch as expected by the model
        #tensor = input_tensor.cpu().detach().numpy()
        #torch.save(tensor,'data/data.raw')

        print("Starting inference...")
        cmd_str="/opt/qti-aic/exec/qaic-runner --test-data "+model_bin +" --write-output-dir "+output_dir +" --aic-device-id "+str(aic)+" --aic-num-of-activations 1  --pre-post-processing on   --input-file 'data/data.raw'"
        subprocess.run(cmd_str,shell=True)
    except Exception as e:
        print('Error running inference: %s' % e)




def imagenet_labels():
    ''' Return a list of imagenet_labels '''
    imagenet_labels = None
    imagenet_index = open('data/imagenet_class_index.json')
    if imagenet_index:
        imagenet_labels = json.load(imagenet_index)
        imagenet_index.close()

    return imagenet_labels

class QAicSoftmax:
    def __init__(self, path, needSoftmax):
        self.softmax = np.fromfile(path, np.float32)
        if (needSoftmax == True):
            self.softmax = np.exp(self.softmax) / np.sum(np.exp(self.softmax), axis=0)

    def shape(self):
        return self.softmax.shape

    def topk(self, k):
        '''Get topk results sorted by highest confidence'''
        topk = {}

        for idx, score in enumerate(self.softmax):
           topk[idx] = score

        if len(topk) == 0:
           return None

        # Sort highest confidence first
        topk = sorted(topk.items(), key=lambda x: x[1], reverse=True)

        return topk[:k] if k else topk[:5]



def pdviewer(binpath, k, needSoftmax=False):
    qaic_softmax = QAicSoftmax(binpath, needSoftmax)
    #print('Softmax dimensions: {}'.format(qaic_softmax.shape()))
    topk = qaic_softmax.topk(k)

    labels = imagenet_labels()
    results = {'Top-K': [], 'Index': [], 'Class': [], 'Confidence': []}

    k_idx = 1
    for idx, confidence in topk:
       if idx==1000:
          continue
       label = labels[str(idx)][1]
       results['Top-K'].append('K{}'.format(k_idx))
       results['Index'].append(idx)
       results['Class'].append(label)
       results['Confidence'].append(confidence)
       k_idx += 1

    print('Top ' + str(k) + ' matches:')
    return pd.DataFrame(results)





if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Processing config input.")
    parser.add_argument("config_file", type=str, help="specify config file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    rtsp_link = config.get("camera_info", "rtsp_link")
    inference_model=config.get("model","inference_model")
    model_name=inference_model
    save_model=load_resnet_50(inference_model)
    model_onnx = pytorch_to_onnx((1,3,256,256), torch.float32, save_model, model_name)
    gen_profile(model_onnx, model_name)
    compile_dir = model_name+ "_prepared"
    convert_to_fp16 = False
    use_int8_quantization_precision = True
    compile_qualcomm(model_onnx, compile_dir, 4, 1, 1, 1, convert_to_fp16, use_int8_quantization_precision)
    validate_qualcomm(compile_dir)
    output_dir = model_name+"_output"
    aic_device = 0

    try:
        cap=cv2.VideoCapture(rtsp_link)
    except ValueError:
        print("Cannot open rtsp link!")

    with open("./imagenet_class_index.json") as json_file:
        d = json.load(json_file)

    image_idx=0
    while True:
        ret,frame=cap.read()

        if not ret:
            break
 #       img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #      image= Image.fromarray(img)
        print("start infenrencing image ",image_idx)
        img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       # print(img)
        #image= Image.fromarray(img)

        prob=predict_qualcomm(img,compile_dir,output_dir,aic_device)
        df= pdviewer(output_dir+'/495-activation-0-inf-0.bin', 3, True)
        print(df, end='\n\n')



        plt.imshow(img)
        plt.axis('off')
        out = df['Class'].loc[0],df['Confidence'].loc[0]
        plt.title(out)
        image_idx += 1

