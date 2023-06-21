import os
import torch
import pickle
import numpy as np
from PIL import Image
import torch.nn as nn
from generator import G_NET
from random import randrange
from encoder import RNN_ENCODER
from discriminator import D_NET256
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
from utils import mkdir_p, display_images

####Configerations Part###
Global_Batch_size = 1
DATA_DIR = "../data/birds"
EMBEDDING_DIM = 256
DF_DIM = 64
NET_G = "netG_epoch_600.pth"
NET_D = "netD2.pth"
WORDS_NUM = 18
RNN_TYPE = "LSTM"
NET_E = "text_encoder599.pth"
B_DCGAN = False
GF_DIM = 32
CONDITION_DIM = 100
BRANCH_NUM = 3
R_NUM = 2
Z_DIM = 100
CUDA = False
FONT_MAX = 50
custome_input = False


def init_word2idx():
    with open("word2idx.pickle", "rb") as handle:
        wordtoix = pickle.load(handle)
    n_words = len(wordtoix)
    return wordtoix, n_words

def init_text_encoder(n_words):
    text_encoder = RNN_ENCODER(n_words, WORDS_NUM, RNN_TYPE, nhidden=EMBEDDING_DIM)
    state_dict = torch.load(NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()
    return text_encoder

def init_generator():
    netG = G_NET( GF_DIM, EMBEDDING_DIM, CONDITION_DIM, Z_DIM, BRANCH_NUM, R_NUM)
    model_dir = NET_G
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

def init_discriminator():
    netD = D_NET256(DF_DIM, EMBEDDING_DIM)
    model_dir = NET_D
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
    netD.load_state_dict(state_dict)
    netD.eval()
    return netD


def gen_example(data_dic, n_words):
    image_paths = []
    text_encoder = init_text_encoder(n_words)
    netG = init_generator()
    s_tmp = "./gen_dir"
    mkdir_p(s_tmp)
    copyofFakeImages = []
    for key in data_dic:
        save_dir = "%s/%s" % (s_tmp, key)
        mkdir_p(save_dir)
        captions, cap_lens = data_dic[key]
        batch_size = captions.shape[0]
        # batch_size = 1
        nz = Z_DIM
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
        for i in range(1):  # 16
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise.data.normal_(0, 1)
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            mask = captions == 0
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            copyofFakeImages.append(fake_imgs[2])
            for j in range(batch_size):
                save_name = "%s/%d_s_" % (save_dir, i)
                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = "%s_g%d.png" % (save_name, k)
                    im.save(fullpath)
                    image_paths.append(fullpath)
    return copyofFakeImages, image_paths


def the_main(input_captions):
    wordtoix, n_words = init_word2idx()

    data_dic = {}  # dictionary used to generate images from captions
    cap_lens = []  # caption lengths
    tokenizer = RegexpTokenizer(r"\w+")
    rev = []
    captions = []
    for input_caption in input_captions:
        tokens = tokenizer.tokenize(input_caption.lower())
        for t in tokens:
            # t = t.encode("ascii", "ignore").decode("ascii")
            rev.append(wordtoix[t])
        captions.append(rev)  # all captions in the file
        cap_lens.append(len(rev))


    
    max_len = np.max(cap_lens)  # used to pad shorter captions
    cap_lens = np.asarray(cap_lens)
    cap_array = np.zeros((len(captions), max_len), dtype="int64")  # placeholder for the padded sorted array caption

    for i in range(len(captions)):
        cap = captions[i]
        c_len = len(cap)
        cap_array[i, : c_len] = cap
    
    for i, input_caption in enumerate(input_captions):
        key = f'caption_{i}'
        cap = np.asanyarray([cap_array[i]])
        # cap_len = np.asanyarray([cap_lens[i]])
        cap_len = np.asanyarray([max_len])
        data_dic[key] = [cap, cap_len]


    f_images, image_paths = gen_example(data_dic, n_words)
    return f_images, image_paths


def discriminator_loss(netD, fake_imgs):
    fake_labels = Variable(torch.FloatTensor(Global_Batch_size).fill_(0))
    print(fake_imgs.shape)
    fake_features = netD(fake_imgs)
    if netD.UNCOND_DNET is not None:
        fake_logits = netD.UNCOND_DNET(fake_features)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = fake_errD
    return errD


def GenerateImages(Input_Captions=["This bird has red wings white belly"], max_tries=10):
    CurrentLoss = [10 for i in range(len(Input_Captions))]
    netD = init_discriminator()
    num_tries = 0
    while (sum(CurrentLoss) > len(Input_Captions) * 0.8) and (num_tries < max_tries):
        num_tries += 1
        print("Try Number", num_tries)
        fake_images, image_paths = the_main(Input_Captions)
        for i, fake_image in enumerate(fake_images):
            dloss = discriminator_loss(netD, fake_image)
            CurrentLoss[i] = dloss.item()
        print("The Loss", CurrentLoss, "The Sum", sum(CurrentLoss), "The Average", sum(CurrentLoss) / len(CurrentLoss))
    return image_paths


if __name__ == "__main__":
    sample_captions = [
        "This bird has red wings white belly",
        "This bird has black head and white belly",
        "This bird has long beak and white belly",
        "This is a blue bird with blue wings and blue belly",
        ]
    GenerateImages(sample_captions)