import ImageBind.data as data
import llama
import json
import torch
import pandas as pd
import cv2
import fundus_prep as prep
# import BERTSimilarity.BERTSimilarity as bertsimilarity
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

def capitalize_sentences(text):
    # if not text[0].isalpha():
    #     text = text[1:]
    sentences = text.split('. ')  # Split the text into sentences

    capitalized_sentences = []
    for sentence in sentences:
        capitalized_sentence = sentence.capitalize()  # Capitalize the first letter
        capitalized_sentences.append(capitalized_sentence)

    capitalized_text = '. '.join(capitalized_sentences)  # Join the sentences back into text
    return capitalized_text

def metrics(preds, labels):
    t_list = [0.5]
    for t in t_list:
        predictions = preds.tolist()
        groundTruth = labels.tolist()
        confusion = confusion_matrix(groundTruth, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        acc = accuracy_score(groundTruth, predictions)
        kappa = cohen_kappa_score(groundTruth, predictions)
        
        from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score
        precision = TP / float(TP+FP)
        sensitivity = TP / float(TP+FN)
        specificity = TN / float(TN+FP)
        F1 = f1_score(groundTruth, predictions)
        balanced_accuracy = balanced_accuracy_score(groundTruth, predictions)
        
        if np.array(groundTruth).max() > 1:
            auc = roc_auc_score(labels, preds, multi_class='ovr')
        else:
            auc = roc_auc_score(labels, preds)
        # print(list(groundTruth), list(predictions))
        print('Threshold:%.4f\tAccuracy:%.4f\tBalanced_Accuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f' % (
            t, acc, balanced_accuracy, sensitivity, specificity, precision, F1, auc, kappa))
        print('TN: %d\t FN:%d\t TP: %d\t FP: %d\n' % (TN, FN, TP, FP))
        return acc, sensitivity, specificity, precision,  F1, auc, kappa

## CUDA_VISIBLE_DEVICES=0,1
def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    448, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            # try:
            #     img = prep.imread(fopen)
            #     r_img, borders, mask, r_img = prep.process_without_gb(img,img)
            #     image = Image.fromarray(cv2.cvtColor(r_img,cv2.COLOR_BGR2RGB))
            # except:
            #     image = Image.open(fopen).convert('RGB')

            # try:
            #     bbox = image.getbbox()
            #     image = image.crop(bbox)
            # except:
            #     pass
            # image = Image.open(filename).convert('HSV')
            image = ImageEnhance.Contrast(image)
            image = image.enhance(1.3)
            image = np.array(image)
            min_R = np.min(image[:,:,0])
            min_G = np.min(image[:,:,1])
            min_B = np.min(image[:,:,2])
            image[:,:,0] = image[:,:,0] - min_R +1
            image[:,:,1] = image[:,:,1] - min_G +1
            image[:,:,2] = image[:,:,2] - min_B +1
            image = Image.fromarray(image.astype('uint8')).convert('HSV')

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)

llama_dir = "/home/ubuntu" # /path/to/LLaMA/

device = "cuda" if torch.cuda.is_available() else "cpu"
model = llama.load("/home/ubuntu/checkpoint-VisionUniteV1.pth", llama_dir)
print(model)

model.to(device)
model.eval()

images = ['/home/ubuntu/normal.jpg']
prompt = ['Describe this image']

input = load_and_transform_vision_data(images, device)
# prompt = prompt.to(device)
results, cls_pred = model.generate(input, prompt, input_type="vision")
print(results)
