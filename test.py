import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import pandas as pd
import numpy

trans = transforms.Compose([transforms.ToTensor()])
"""img = Image.open("img/IMG_3231.jpg")
trans_img = trans(img)
print(trans_img)
#print(trans_img.size())"""
#print(trans_img.shape)

real_rg = [None] * 3
ai_rg = [None] * 3
real_avg = [None] * 3
ai_avg = [None]*3

# timgs is an array of training images and the real/AI labels
def train(timgs):
    global real_rg
    global ai_rg
    global real_avg
    global ai_avg
    #for i in range(len(timgs)):
    for i in range(2000):
        img = Image.open(timgs[i][1])
        trans_img = trans(img)
        if len(trans_img) >= 3:
            red_rg = torch.max(trans_img[0]) - torch.min(trans_img[0])
            gr_rg = torch.max(trans_img[1]) - torch.min(trans_img[1])
            bl_rg = torch.max(trans_img[2]) - torch.min(trans_img[2])

            if timgs[i][2] == 0 and real_rg[0] == None:
                real_rg[0] = red_rg
                real_rg[1] = gr_rg
                real_rg[2] = bl_rg
            elif timgs[i][2] == 0:
                real_rg[0] = (real_rg[0] + red_rg)/2
                real_rg[1] = (real_rg[1] + gr_rg)/2
                real_rg[2] = (real_rg[2] + bl_rg)/2
            elif timgs[i][2] == 1 and ai_rg[0] == None:
                ai_rg[0] = red_rg
                ai_rg[1] = gr_rg
                ai_rg[2] = bl_rg
            else:
                ai_rg[0] = (ai_rg[0] + red_rg)/2
                ai_rg[1] = (ai_rg[1] + gr_rg)/2
                ai_rg[2] = (ai_rg[2] + bl_rg)/2

            ch_avg = torch.mean(trans_img, dim=(1,2))
            if timgs[i][2] == 0 and real_avg[0] == None:
                real_avg = ch_avg
            elif timgs[i][2] == 0:
                real_avg[0] = (real_avg[0] + ch_avg[0])/2
                real_avg[1] = (real_avg[1] + ch_avg[1])/2
                real_avg[2] = (real_avg[2] + ch_avg[2])/2
            elif timgs[i][2] == 1 and ai_avg[0] == None:
                ai_avg = ch_avg
            else:
                ai_avg[0] = (ai_avg[0] + ch_avg[0])/2
                ai_avg[1] = (ai_avg[1] + ch_avg[1])/2
                ai_avg[2] = (ai_avg[2] + ch_avg[2])/2

def identify(img_path):
    global real_rg
    global ai_rg
    global real_avg
    global ai_avg
    real = 0
    ai = 0
    img = Image.open(img_path)
    trans_img = trans(img)
    if len(trans_img) >= 3:
        red_rg = torch.max(trans_img[0]) - torch.min(trans_img[0])
        gr_rg = torch.max(trans_img[1]) - torch.min(trans_img[1])
        bl_rg = torch.max(trans_img[2]) - torch.min(trans_img[2])
        if abs(red_rg - real_rg[0].item()) <= abs(red_rg - ai_rg[0].item()):
            real += 1
        else:
            ai += 1
        if abs(gr_rg - real_rg[1].item()) <= abs(gr_rg - ai_rg[1].item()):
            real += 1
        else:
            ai += 1
        if abs(bl_rg - real_rg[2].item()) <= abs(bl_rg - ai_rg[2].item()):
            real += 1
        else:
            ai += 1

        ch_avg = torch.mean(trans_img, dim=(1,2))
        if abs(ch_avg[0].item() - real_avg[0].item()) < abs(ch_avg[0].item() - ai_avg[0].item()):
            real += 1
        else:
            ai += 1
        if abs(ch_avg[1].item() - real_avg[1].item()) < abs(ch_avg[1].item() - ai_avg[1].item()):
            real += 1
        else:
                ai += 1
        if abs(ch_avg[2].item() - real_avg[2].item()) < abs(ch_avg[2].item() - ai_avg[2].item()):
            real += 1
        else:
            ai += 1

        if real > ai:
            return 0
        elif ai > real:
            return 1
        else:
            return numpy.random.choice([0, 1])
            


df = pd.read_csv("train.csv")  
train_arr = df.values  

train(train_arr)
print(identify("img/ai.jpg"))