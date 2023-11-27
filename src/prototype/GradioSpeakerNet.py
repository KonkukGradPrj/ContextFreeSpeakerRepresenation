import sys, time, os, argparse, socket
import numpy as np
from scipy.io import wavfile
import torch
from SpeakerNet import SpeakerNet
import torchaudio

class model_for_gradio():
    def __init__(self,input_file,enrolled_file):
        self.input_file = input_file
        self.enrolled_file = enrolled_file
        self.model = SpeakerNet(lr = 0.0001, model="ResNetSE34L", nOut = 512, encoder_type = 'SAP', normalize = True, unif_loss='uniform', sim_loss='anglecontrast', lambda_u=1, lambda_s=1, t=2, sample_type='PoN');

    def load_weight(self,weight_path):
        self.model.loadParameters(weight_path)
        print("Model %s loaded!"%weight_path);

    def is_enrolled(self,Threshold):
        ret_dict = dict()
        
        feats,cos_sim = self.model.get_embedding(self.input_file,self.enrolled_file,eval_frames=0)
        
        cos_sim = (cos_sim - 0.6) * 5
        ret_dict['cos_sim'] = cos_sim
        if cos_sim >= Threshold:
            ret_dict['is_enroll'] = True
        else:
            ret_dict['is_enroll'] = False
        
        return ret_dict

if __name__ == '__main__':
    a = wavfile.read('/home/hyeons/workspace/DB/vox1_test/wav/id10270/GWXujl-xAVM/00036.wav')
    
    embed = np.array([ 3.3106e-02,  1.7731e-02, -8.5648e-02, -8.4468e-02,  4.5231e-03,
         -3.8563e-02,  3.5099e-02,  4.4115e-02, -6.8899e-02,  9.9147e-02,
          1.4543e-02, -3.5797e-02,  1.0495e-02, -1.7602e-02,  3.1585e-02,
          3.1386e-02,  2.6977e-02, -9.8218e-03,  9.1477e-03,  7.7244e-02,
         -1.7644e-02, -3.5009e-02,  6.1358e-02,  5.6377e-02, -1.7482e-02,
         -4.7887e-03, -6.7311e-03,  2.5456e-02,  3.7556e-02,  1.6851e-02,
         -3.9296e-02, -3.2488e-02,  3.4753e-02, -2.3213e-02,  2.0601e-02,
         -3.5780e-02,  1.6816e-02,  2.9106e-02, -2.7286e-02, -7.0690e-03,
         -1.1547e-02,  6.4395e-02, -1.9359e-02,  6.9771e-03,  2.0381e-03,
         -1.3454e-02, -1.4742e-02, -9.5718e-02,  9.3942e-03, -8.7095e-03,
         -2.5995e-02, -2.0086e-03, -9.1677e-02, -2.6022e-02,  3.4911e-02,
          1.3197e-02, -2.1144e-02,  3.8549e-02, -3.1998e-02,  5.2974e-02,
          1.1739e-02, -6.8862e-02,  5.2180e-02, -9.7953e-03, -6.2999e-02,
          1.5858e-02,  5.4579e-02, -8.7204e-02, -2.5143e-02, -4.4643e-02,
          2.2687e-02,  6.3602e-02, -2.2121e-02,  3.7052e-02, -1.5706e-02,
          1.6065e-02, -5.9324e-03,  1.8129e-02, -4.7147e-02, -4.0503e-03,
         -4.9172e-02, -3.6950e-03,  1.5316e-02,  1.4814e-02, -1.0811e-02,
          3.8139e-03,  5.6767e-02, -2.2111e-03,  7.5549e-03, -6.6104e-03,
         -1.6730e-02,  8.7611e-02, -3.5843e-02, -2.1505e-02, -3.5422e-02,
          3.6682e-02,  4.5753e-02,  3.5201e-02,  3.7190e-02, -4.4312e-02,
          2.8846e-02,  2.0584e-02,  9.4311e-02,  5.0880e-02, -2.4582e-02,
          5.1678e-02, -1.1953e-01,  4.4358e-03, -2.8930e-02,  1.1046e-02,
         -2.2931e-03,  1.6633e-03, -2.4607e-02,  8.0055e-02, -5.9177e-02,
         -4.9312e-02,  5.5396e-02, -3.3421e-02,  3.5354e-02,  2.5700e-02,
         -5.4223e-02, -5.9809e-03,  3.7289e-02, -2.9816e-02,  4.3657e-02,
         -4.2319e-02,  3.5401e-02, -7.4780e-02, -5.5951e-03,  5.3530e-02,
          3.4566e-02, -2.2562e-03,  6.5768e-02,  5.7736e-02, -2.1565e-02,
          2.4648e-04, -1.1782e-02,  3.5237e-02, -1.8308e-02, -2.0309e-02,
         -3.3667e-02,  5.7284e-02, -6.7161e-02,  1.1995e-02,  1.0619e-02,
         -2.0200e-02, -2.8481e-02, -1.9808e-02, -2.5126e-02,  7.4005e-02,
          3.8506e-02, -3.8756e-02,  4.5363e-02,  5.9150e-02, -2.8238e-02,
         -2.9826e-02, -2.7746e-02,  5.8400e-02, -5.0231e-02,  4.0548e-02,
         -1.4383e-02,  1.3586e-02, -6.6819e-02,  3.6962e-03,  3.6914e-02,
          4.3118e-02, -1.3799e-02,  1.2716e-02,  1.7808e-02, -3.9658e-02,
         -4.7639e-03, -5.7115e-03, -2.8867e-02, -1.3030e-02,  8.4148e-02,
         -1.8866e-02,  8.3103e-02,  2.9411e-02, -2.1193e-02,  2.3524e-02,
         -4.6055e-02, -6.0215e-03, -6.3775e-02, -5.4047e-02, -1.3963e-02,
         -4.3810e-03,  3.4882e-02, -6.5781e-02,  2.8515e-03, -9.9560e-03,
          4.8799e-02,  6.1531e-02,  3.6417e-02,  1.8114e-03,  3.3426e-02,
          3.1085e-02, -5.8463e-02, -8.0988e-02, -1.7303e-02,  1.3465e-02,
          3.4051e-02, -5.3678e-02, -2.8692e-03,  5.7349e-02, -2.6532e-02,
          2.0410e-02, -5.6471e-02,  4.0382e-02, -7.7058e-02,  5.3651e-02,
         -1.2400e-02,  7.5172e-02,  1.8026e-02, -3.9731e-02,  5.0372e-02,
          2.1026e-02, -4.5962e-02, -1.6285e-02, -3.8801e-02, -2.3031e-02,
          4.8827e-02, -1.8694e-02, -4.3770e-02, -3.2945e-03, -6.8927e-02,
         -6.2627e-02,  7.6618e-02, -2.9767e-03,  8.9653e-02,  3.9928e-02,
          5.3936e-02, -4.0011e-03,  5.3952e-02,  1.4868e-02,  6.0831e-04,
          4.9249e-02,  2.7805e-02,  3.3822e-02, -8.7184e-02,  4.9792e-02,
          5.4193e-02,  2.0543e-02, -2.3294e-02,  7.8537e-02,  4.9349e-03,
         -1.4841e-02, -3.9011e-02, -7.5624e-02,  1.4985e-02,  2.8884e-02,
         -5.6981e-02,  2.6282e-02,  4.7288e-02, -1.7718e-02,  3.7581e-02,
          8.6972e-03, -6.3468e-03, -6.2122e-02, -5.2795e-02,  5.7514e-02,
          4.9999e-02,  6.8254e-03, -5.1235e-04,  3.0891e-02, -2.9628e-03,
          8.7738e-02, -6.7942e-03,  1.1801e-02,  1.6441e-02,  2.2028e-02,
          6.0466e-02, -2.7211e-02, -1.0008e-01,  1.1065e-02,  3.1320e-02,
          5.7144e-02,  7.6360e-02,  4.2242e-03,  2.7922e-03,  1.4758e-02,
          2.0910e-03, -1.7356e-02, -3.1135e-02, -4.3763e-02, -7.1551e-02,
          1.0067e-02, -2.4310e-02, -5.6898e-03, -3.4674e-02,  6.4591e-02,
         -9.1387e-02, -6.9636e-02, -5.5966e-03,  5.4950e-02, -2.5384e-02,
         -3.3361e-02,  5.4405e-02, -6.4040e-02,  4.7416e-02,  1.7311e-03,
          7.6846e-03, -7.3820e-03,  4.5089e-02, -3.5605e-02, -3.1693e-03,
         -4.7766e-03, -9.8491e-02,  2.2635e-02,  5.9120e-02,  5.4582e-02,
         -7.2642e-02, -4.3118e-02,  5.3047e-02, -5.6492e-03, -1.6709e-02,
         -4.7546e-02,  3.5752e-02,  1.5707e-02, -4.7080e-03, -7.6482e-03,
          1.3462e-02, -2.4049e-02, -4.2540e-02,  2.0532e-02, -1.5221e-03,
         -1.7543e-02, -4.7504e-03, -5.8708e-03, -1.4798e-02,  2.5196e-02,
         -3.6043e-03, -4.6266e-02,  1.1639e-02, -9.0598e-02,  8.6521e-02,
         -7.0487e-02,  3.9060e-02, -1.1843e-02,  9.8189e-03,  3.2670e-02,
         -5.5915e-02,  2.8262e-02,  1.7704e-02, -1.2088e-01, -1.2761e-02,
         -7.2463e-02, -3.6386e-03, -2.0938e-02,  3.8915e-02,  1.6400e-02,
         -8.0395e-03,  8.0571e-03,  3.1862e-02,  8.6694e-04,  5.6244e-03,
         -3.1303e-02,  6.1771e-02,  4.2620e-03, -8.0512e-02, -6.8235e-02,
         -2.2318e-02, -4.7065e-02,  6.9285e-02,  1.3795e-02,  2.2813e-02,
         -8.2532e-02, -1.1691e-02, -4.4621e-05, -5.4009e-03,  8.1470e-02,
          5.6131e-02, -3.1347e-02,  3.6043e-02, -9.4113e-02,  9.1423e-02,
          4.3937e-02, -8.6610e-02,  3.4622e-03,  5.7562e-02, -2.1850e-02,
          3.0665e-02, -1.8883e-02, -4.3069e-02, -2.3976e-02, -3.5155e-02,
          9.3608e-04,  3.6088e-02,  1.1833e-02,  4.8136e-02, -9.0008e-02,
         -3.1184e-02,  7.8706e-02,  2.1486e-02, -2.4054e-02, -5.6751e-02,
         -9.5535e-02, -5.7776e-02, -6.6023e-02, -3.4419e-02, -3.9943e-02,
         -2.6108e-02, -6.5288e-02,  2.9117e-02, -1.4251e-02,  5.5259e-02,
          1.2966e-03,  5.4229e-02,  6.2243e-02, -3.5309e-02, -2.4043e-02,
          3.2821e-02,  2.5154e-02,  7.7021e-02,  4.1145e-02, -6.9396e-02,
         -6.5745e-02, -1.3985e-01,  2.0869e-02, -1.6707e-04, -5.4604e-02,
         -5.5252e-03,  1.4311e-02,  7.3091e-03, -1.3066e-02,  2.0236e-02,
         -7.9292e-02,  5.4361e-02, -3.8266e-03, -7.1608e-02, -4.2903e-02,
          4.5153e-02,  1.0133e-02, -3.0302e-02, -6.0409e-02,  4.8913e-02,
          5.3501e-03, -6.1401e-02, -5.6278e-02,  4.8316e-02,  2.5538e-02,
         -7.5360e-02, -3.2091e-02,  3.9456e-02,  3.5460e-02,  1.6816e-02,
         -3.0733e-02,  6.4393e-02, -9.0956e-03, -9.8078e-02,  4.9326e-02,
          1.1564e-02, -2.5254e-02, -4.4315e-03, -5.4968e-02, -2.7441e-02,
         -3.8137e-02,  1.0250e-01, -7.1238e-03, -2.7580e-02, -2.1332e-02,
          2.9310e-02, -3.9532e-02, -1.3900e-02, -3.5603e-02,  2.1872e-02,
         -3.8640e-02, -2.8391e-02, -5.9570e-03, -2.7107e-02,  5.3627e-02,
         -8.3432e-02, -2.1154e-02,  8.2312e-02,  7.3213e-02, -6.1846e-02,
         -2.8926e-02, -5.1326e-02, -8.3588e-02,  4.5074e-02, -1.3705e-02,
         -2.0410e-04, -5.5451e-02, -3.4409e-02, -1.3812e-02, -6.3827e-02,
         -1.0943e-01,  5.6764e-02, -1.7488e-03, -4.4994e-03, -2.4484e-02,
         -4.7788e-02,  6.3568e-02, -6.0986e-02,  3.9575e-02, -3.5742e-03,
          2.1916e-02,  9.5377e-03,  2.1774e-03,  2.0016e-02,  3.2515e-02,
         -2.8161e-02, -9.7666e-02, -1.5434e-02, -3.1675e-03, -5.5485e-02,
          1.0161e-02, -3.4630e-02,  6.0707e-02,  2.1829e-02, -4.2626e-02,
         -5.3070e-02,  3.8121e-02])
    
    gradio = model_for_gradio(a[1],embed)
    gradio.load_weight('/home/hyeons/workspace/ContextFreeSpeakerRepresenation/src/baseline/contrastive-equilibrium-learning/save/pre-trained_unspv_unif-a-prot.model')
    
    print(gradio.is_enroll(0.71))