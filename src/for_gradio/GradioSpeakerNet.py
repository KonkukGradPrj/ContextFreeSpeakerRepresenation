import sys, time, os, argparse, socket
import numpy
import torch
from SpeakerNet import SpeakerNet

class model_for_gradio():
    def __init__(self,input_path,enrolled_path,root_path):
        self.input_path = input_path
        self.enrolled_path = enrolled_path
        self.root_path = root_path

        self.model = SpeakerNet(lr = 0.0001, model="ResNetSE34L", nOut = 512, encoder_type = 'SAP', normalize = True, unif_loss='uniform', sim_loss='anglecontrast', lambda_u=1, lambda_s=1, t=2, sample_type='PoN');

    def load_weight(self,weight_path):
        self.model.loadParameters(weight_path)
        print("Model %s loaded!"%weight_path);

    def is_enroll(self,Threshold):
        feats,cos_sim = self.model.get_embedding(input_path=self.input_path,enrolled_path=self.enrolled_path,root_path=self.root_path,eval_frames=0)
        print(cos_sim)
        if cos_sim >= Threshold:
            return True
        else:
            return False

if __name__ == '__main__':
    gradio = model_for_gradio('id10270/GWXujl-xAVM/00035.wav',"id10270/zjwijMp0Qyw/00001.wav",'/home/hyeons/workspace/DB/vox1_test/wav/')

    gradio.load_weight('/home/hyeons/workspace/ContextFreeSpeakerRepresenation/src/baseline/contrastive-equilibrium-learning/save/pre-trained_unspv_unif-a-prot.model')
    print(gradio.is_enroll(0.71))
    