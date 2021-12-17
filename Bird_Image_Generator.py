#!/usr/bin/env python
# coding: utf-8

# In[1]:


####Configerations Part###
Global_Batch_size = 1
DATA_DIR = "../data/birds"
EMBEDDING_DIM = 256
DF_DIM = 64
NET_G = "netG_epoch_600.pth"
NET_D = "netD2.pth"
WORDS_NUM = 18
RNN_TYPE = 'LSTM'
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

###########

###Imports Part###
import os
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from PIL import Image
import errno
import warnings
warnings.filterwarnings('ignore')


# In[2]:


############
###utils functions###

def display_images(bs=1, currentDirectory="/"):
  filepath = "example_captions.txt"
  fileNamesList = []
  dst = os.path.join(currentDirectory, "assets")
  with open(filepath, "r") as f:
    sentences = f.read().split('\n')
  for k in range(bs):
    filename0 = './content/Bird-Image-Generator/netG_epoch_600/example_captions/0_s_%s_g0.png' %k
    filename1 = './content/Bird-Image-Generator/netG_epoch_600/example_captions/0_s_%s_g1.png' %k
    filename2 = './content/Bird-Image-Generator/netG_epoch_600/example_captions/0_s_%s_g2.png' %k

    from shutil import copy
    copy(filename2, dst)
    fileNamesList.append(filename2)
    # display(z)

  os.chdir(currentDirectory)
  return fileNamesList
#     display(x, y ,z)



def gen_example(data_dic, n_words):
    # Build and load the generator
    #print ("10. creating an object of RNN_ENCODER")
    text_encoder = RNN_ENCODER(n_words, nhidden=EMBEDDING_DIM)
    #print("15. text_encoder object of RNN_ENCODER created")
    state_dict = torch.load(NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    #print("16. Loaded weights for text_encoder from:", NET_E)
    #cudaaa text_encoder = text_encoder.cuda()
    text_encoder.eval()
    #print("17. text_encoder moved to gpu and in evaluation mode")

    # the path to save generated images
    if B_DCGAN:
        netG = 3#G_DCGAN()
    else:
        #print("18. creating an object of G_NET (Generator)")
        netG = G_NET()
        #print("47. object of G_NET (netG) 'Three stage Generator created'")
    s_tmp = NET_G[:NET_G.rfind('.pth')]
    model_dir = NET_G

    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    #print("48. Loaded weights for netG from:", model_dir)
    #cudaaa netG.cuda()
    netG.eval()


    #print("49. text_encoder moved to gpu and in evaluation mode")
    for key in data_dic:
        s_tmp = './content/Bird-Image-Generator/netG_epoch_600'
        save_dir = '%s/%s' % (s_tmp, key)
        mkdir_p(save_dir)
        #print ("50. Created directory with name : ", save_dir)
        captions, cap_lens, sorted_indices = data_dic[key]
        
        #print("51. fetched captions, cap_lens, sorted_indices from the dictionary")

        batch_size = captions.shape[0]
        Global_Batch_size = batch_size
        print("***Global Batch Size : ", Global_Batch_size)
        #print("52. got batch size from caption array (number of sentences in file) as :" ,batch_size)
        nz = Z_DIM
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

        #cudaaa captions = captions.cuda()
        #cudaaa cap_lens = cap_lens.cuda()
        #print("53. Loaded captions to pytorch and moved to cuda")
        for i in range(1):  # 16
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            #cudaaa noise = noise.cuda()
            #print("54. Loaded noise to pytorch and moved to cuda")
            #######################################################
            # (1) Extract text embeddings
            ######################################################
            #print("55. Calling init_hidden of RNN_ENCODER (text_encoder)")
            hidden = text_encoder.init_hidden(batch_size)
            #print("56. finished init_hidden of RNN_ENCODER (text_encoder)")
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            #print("57. Calling forward of RNN_ENCODER (text_encoder)")
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            #print("58. finished forward of RNN_ENCODER (text_encoder)")
            mask = (captions == 0)
            #print("59. something to make mask of captions")
            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            #print ("60. Calling forward of G_NET (netG)")
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            copyofFakeImages = fake_imgs
            #print ("86. finished forward of G_NET (netG)")
            # G attention
            cap_lens_np = cap_lens.cpu().data.numpy()
            for j in range(batch_size):
                save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                #print("87.Saving each image generated of the 16 captions with name: (", save_name,")")
                for k in range(len(fake_imgs)):
                    #print("88. looping through the 3 images from each stage to save (",k+1,"of 3)")
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    #print('89.image number :', k+1 ," with shape: ", im.shape, "moved to cpu as numpy array")
                    im = np.transpose(im, (1, 2, 0))
                    #print('90. image transposed to change to PIL image with shape : ', im.shape)
                    im = Image.fromarray(im)
                    fullpath = '%s_g%d.png' % (save_name, k)
                    im.save(fullpath)
                    #print("91. image saved to path: ", fullpath)

                # for k in range(len(attention_maps)):
                #     if len(fake_imgs) > 1:
                #         im = fake_imgs[k + 1].detach().cpu()
                #     else:
                #         im = fake_imgs[0].detach().cpu()
                #     attn_maps = attention_maps[k]
                #     att_sze = attn_maps.size(2)
                #     img_set, sentences = \
                #         build_super_images2(im[j].unsqueeze(0),
                #                             captions[j].unsqueeze(0),
                #                             [cap_lens_np[j]], wordtoix,
                #                             [attn_maps[j]], att_sze)
                #     if img_set is not None:
                #         im = Image.fromarray(img_set)
                #         fullpath = '%s_a%d.png' % (save_name, k)
                #         im.save(fullpath)
    return copyofFakeImages

def mkdir_p(path):
    #print("using mkdir_p")
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def upBlock(in_planes, out_planes):
    #print("using upBlock")
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    #print("using conv1x1")
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                    padding=0, bias=False)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    #print("using conv3x3")
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

def describe_tensor (name , tensor):
    #print (name ,": size : " , tensor.size())
    #print (name ,": first element : " , tensor[0])
    #print (name ,": non-zeros : ",tensor.view(-1).nonzero().size())
    name = ""


#####################
###Class of Models###

class RNN_ENCODER(nn.Module):
    def __init__(self,
                ntoken,
                ninput=300,
                drop_prob=0.5,
                nhidden=128,
                nlayers=1,
                bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary that maps words to unique indexes
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        #print ("11. calling define_module of RNN_ENCODER")
        self.define_module()
        #print ("13. calling init_weights of RNN_ENCODER")
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput,
                                self.nhidden,
                                self.nlayers,
                                batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
            #print ("12.finshed define_module of RNN_ENCODER")
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput,
                                self.nhidden,
                                self.nlayers,
                                batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        #print ("14. finished init_weights of RNN_ENCODER")

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable( weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                    Variable( weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))
        else:
            return Variable(
                weight.new(self.nlayers * self.num_directions, bsz,
                           self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #print("--------------------------------------")
        describe_tensor("emb" , emb)
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]


        
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        return words_emb, sent_emb

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = EMBEDDING_DIM
        self.c_dim = CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        #print("20. Creating an object of GLU")
        self.relu = GLU()
        #print("21. object of GLU (relu) created")

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if CUDA:
            #cudaaa eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        #print("62. Calling encode of CA_NET")
        mu, logvar = self.encode(text_embedding)
        #print("63. finished encode of CA_NET")
        #print("64. Calling reparamatrize of CA_NET")
        c_code = self.reparametrize(mu, logvar)
        #print("65. finished reparamatrize of CA_NET")
        describe_tensor("text_embedding",text_embedding)
        describe_tensor("mu",mu)
        describe_tensor("logvar",logvar)
        describe_tensor("c_code",c_code)
        return c_code, mu, logvar

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        #print("Using GLU")

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM
        #print("24. calling define module of INIT_STAGE_G ")
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())
        #print("25.Creating 4 upBlock Layers (upBlock: Sequential of: nn.Upsample, conv3x3, nn.BatchNorm2d, GLU)")
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        #print("26.upsample1, upsample2, upsample3 and upsample4 careted ")
        #print("27. finished define module of INIT_STAGE_G")

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        #print("34. make conv_context of conv1x1")
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = R_NUM
        #print("32. calling define_module of NEXT_STAGE_G ")
        self.define_module()
        #print("39. finished define_module of NEXT_STAGE_G")

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        #print("33.Creating an object of GlobalAttentionGeneral")
        self.att = GlobalAttentionGeneral(ngf, self.ef_dim)
        #print("35. object of GlobalAttentionGeneral (att) created")
        #print("36. calling _make_layer of NEXT_STAGE_G (R_NUM=2 of resBlocks) ")
        self.residual = self._make_layer(ResBlock, ngf * 2)
        #print("37. finished _make_layer of NEXT_STAGE_G")
        #print("38. Creating 1 upBlock Layer")
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        #print ("73. Calling applyMask of GlobalAttentionGeneral(att)")
        self.att.applyMask(mask)
        #print ("74. finished applyMask of GlobalAttentionGeneral(att)")
        #print ("75. Calling forward of GlobalAttentionGeneral(att)")
        c_code, att = self.att(h_code, word_embs)
        #print ("76. finished forward of GlobalAttentionGeneral(att)")
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        #print("using ResBlock(Sequential : conv3x3, nn.BatchNorm2d, GLU, conv3x3, nn.BatchNorm2d)")
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = GF_DIM
        nef = EMBEDDING_DIM
        ncf = CONDITION_DIM
        #print("19.Creating an object CA_NET")
        self.ca_net = CA_NET()
        #print("22.object of CA_NET (ca_net) created")

        if BRANCH_NUM > 0:
            #print("23.Creating an object INIT_STAGE_G (first Generator)")
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            #print("28. object of INIT_STAGE_G (h_net1) created")
            #print("29. Creating an object of GET_IMAGE_G")
            self.img_net1 = GET_IMAGE_G(ngf)
            #print("30. object of GET_IMAGE_G (img_net1) created")
        # gf x 64 x 64
        if BRANCH_NUM > 1:
            #print("31.Creating an object NEXT_STAGE_G (second Generator)")
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            #print("40.object of NEXT_STAGE_G(h_net2) created")
            #print("41. Creating an object of GET_IMAGE_G")
            self.img_net2 = GET_IMAGE_G(ngf)
            #print("42. object of GET_IMAGE_G (img_net2) created")
        if BRANCH_NUM > 2:
            #print("43.Creating an object NEXT_STAGE_G (third Generator)")
            #print("--------------BeginDuplicate--------------")
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            #print("---------------EndDuplicate---------------")
            #print("44.object of NEXT_STAGE_G(h_net3) created")
            #print("45. Creating an object of GET_IMAGE_G")
            self.img_net3 = GET_IMAGE_G(ngf)
            #print("46. object of GET_IMAGE_G (img_net3) created")

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        describe_tensor("z_code " , z_code)
        describe_tensor(" sent_emb" ,sent_emb )
        describe_tensor("word_embs" ,word_embs )
        describe_tensor("mask " ,mask )
        fake_imgs = []
        att_maps = []
        #print ("61. Calling forward of CA_NET (ca_net)")
        c_code, mu, logvar = self.ca_net(sent_emb)
        #print ("66. finished forward of CA_NET (ca_net)")

        if BRANCH_NUM > 0:
            #print ("67. Calling forward of INIT_STAGE_G (h_net1)")
            h_code1 = self.h_net1(z_code, c_code)
            #print ("68. finished forward of INIT_STAGE_G (h_net1)")
            #print ("69. Calling forward of GET_IMAGE_G (img_net1)")
            fake_img1 = self.img_net1(h_code1)
            #print ("70. finished forward of GET_IMAGE_G (img_net1)")
            fake_imgs.append(fake_img1)
            #print ("71. appended first stage images (fake_imgs1) to the full list (fake_imgs) ")
        if BRANCH_NUM > 1:
            #print ("72. Calling forward of NEXT_STAGE_G (h_net2)")
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            #print ("77. finshed forward of NEXT_STAGE_G (h_net2)")
            #print ("78. Calling forward of GET_IMAGE_G (img_net2)")
            fake_img2 = self.img_net2(h_code2)
            #print ("79. finished forward of GET_IMAGE_G (img_net2)")
            fake_imgs.append(fake_img2)
            #print ("80. appended second stage images (fake_imgs2) to the full list (fake_imgs) ")
            if att1 is not None:
                att_maps.append(att1)
        if BRANCH_NUM > 2:
            #print ("81. Calling forward of NEXT_STAGE_G (h_net3)")
            #print("--------------BeginDuplicate--------------")
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            #print("---------------EndDuplicate---------------")
            #print ("82. finshed forward of NEXT_STAGE_G (h_net3)")
            #print ("83. Calling forward of GET_IMAGE_G (img_net3)")
            fake_img3 = self.img_net3(h_code3)
            #print ("84. Calling forward of GET_IMAGE_G (img_net3)")
            fake_imgs.append(fake_img3)
            #print ("85. appended third stage images (fake_imgs3) to the full list (fake_imgs) ")
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar

def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)

class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        ndf = DF_DIM
        nef = EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4

def the_main():

  #with open('wordtoix.pickle', 'rb') as handle:
      #wordtoix = pickle.load(handle)

  
  print("||||||||||||||||||||||||||||||||||| WORD TO INDEX ||||||||||||||||||||||||||||||||||||||||")
  wordtoix = {'fawn': 1, 'striple': 2109, 'blsck': 2, 'reds': 4263, 'smallperching': 3, 'waterbird': 4300, 'headand': 5, 'mandible': 6, 'squared': 1805, 'yellos': 8, 'comparatively': 9, 'dynamic': 10, 'all': 2728, 'yellow': 12, 'sleek': 13, 'four': 14, 'rapid': 5134, 'sleep': 16, 'hanging': 17, 'consists': 18, 'woody': 19, 'ghroat': 5384, 'comically': 20, 'adorning': 22, 'whose': 24, 'feeding': 25, 'patches': 26, 'eligible': 27, 'ventral': 28, 'segments': 29, 'swan': 30, 'undertails': 34, 'patched': 35, 'bike': 36, 'cactus': 3193, 'buugy': 38, 'under': 39, 'regal': 40, 'downcurved': 41, 'immature': 42, 'adabomin': 43, 'thoart': 44, 'smallbird': 246, 'alternating': 47, 'supercilary': 3657, 'risk': 49, 'stright': 50, 'rise': 53, 'connects': 55, 'every': 59, 'foul': 60, 'affect': 357, 'interleaved': 64, 'look': 3660, 'wooded': 65, 'dominate': 1814, 'coverst': 5085, 'posessing': 2829, 'parrot': 68, 'throated': 2739, 'wooden': 70, 'pinch': 71, 'clings': 72, 'topside': 3178, 'solid': 3662, 'flutters': 75, 'multicolored': 5346, 'greenish': 76, 'nondescript': 77, 'shows': 1817, 'undercover': 79, 'fluttery': 80, 'glassy': 581, 'sooty': 82, 'feathery': 83, 'eagle': 1818, 'practically': 4085, 'consistent': 85, 'ornate': 86, 'feathers': 87, 'lavender': 88, 'frosted': 89, 'horn': 90, 'nail': 91, 'surrounding': 92, 'rusty': 93, 'fabulous': 94, 'lines': 4427, 'winder': 96, 'tether': 97, 'scraped': 620, 'colourful': 99, 'specs': 100, 'virbrant': 101, 'shining': 102, 'blue': 103, 'expanded': 3665, 'hide': 105, 'semicircular': 106, 'ugly': 107, 'creating': 4887, 'yelloe': 108, 'striations': 109, 'throatpatch': 110, 'tipe': 111, 'lizard': 112, 'asia': 113, 'lights': 115, 'windbags': 4478, 'wingpsan': 706, 'supercialiary': 117, 'above': 118, 'blur': 119, 'lond': 4604, 'hsort': 729, 'tips': 121, 'targuses': 2747, 'ever': 122, 'consisting': 123, 'harbors': 124, 'submerged': 4255, 'textures': 125, 'med': 126, 'bi': 3893, 'camoflaged': 4566, 'screams': 128, 'chins': 129, 'festive': 130, 'avery': 131, 'shimmery': 133, 'neckline': 134, 'wrapping': 135, 'whiteish': 136, 'here': 137, 'natural': 1665, 'beck': 139, 'protection': 140, 'songbirds': 141, 'textured': 144, 'path': 146, 'stares': 147, 'darkly': 4568, 'curled': 2055, 'speckeld': 149, 'either': 3886, 'hairdo': 151, 'multicoloed': 152, 'balance': 153, 'uplifting': 154, 'spottd': 155, 'taper': 156, 'rests': 157, 'smoky': 158, 'piercing': 159, 'complimeted': 160, 'robin': 3678, 'winglets': 5213, 'drk': 163, 'ancestry': 4572, 'tortoiseshell': 164, 'surepciliary': 165, 'smoke': 166, 'natures': 167, 'symmetrical': 168, 'suitable': 169, 'changes': 170, 'midway': 171, 'colred': 1830, 'eyebrows': 173, 'appropriately': 1019, 'fantastic': 175, 'itsback': 176, 'intensely': 1590, 'stray': 178, 'lengthed': 179, 'straw': 180, 'spooky': 181, 'supercillery': 5074, 'predominant': 5414, 'brs': 183, 'visibly': 184, 'orangish': 185, 'visible': 186, 'cheats': 187, 'eyering': 2755, 'stern': 771, 'perhaps': 5057, 'glance': 191, 'total': 192, 'curiously': 2614, 'eyebrown': 194, 'ebony': 195, 'slivers': 196, 'plummish': 2976, 'meringue': 5123, 'would': 198, 'army': 199, 'remainder': 1834, 'spiky': 201, 'overshadow': 202, 'speckels': 203, 'pudgy': 204, 'sidebars': 1202, 'ellow': 206, 'overhead': 207, 'calm': 208, 'spike': 209, 'strike': 210, 'airbound': 3946, 'downcurving': 212, 'type': 213, 'until': 214, 'separated': 216, 'cocoa': 5293, 'females': 219, 'th': 2760, 'tarses': 221, 'scary': 4579, 'clearly': 4706, 'reletively': 223, 'curvature': 224, 'lore': 225, 'warm': 226, 'adult': 228, 'separates': 229, '90': 230, 'ward': 231, 'hold': 232, 'pouch': 233, 'shart': 5182, 'must': 235, 'me': 236, 'eeverywere': 2765, 'reed': 3980, 'ceown': 241, 'mo': 242, 'blade': 243, 'worm': 244, 'roof': 245, 'spec': 45, 'backside': 1841, 'erg': 248, 'poitned': 249, 'appendage': 250, 'elegantly': 252, 'my': 254, 'example': 255, 'carmel': 256, 'pinecone': 257, 'decorative': 258, 'shest': 259, 'unmarked': 262, 'indicated': 263, 'give': 264, 'grewy': 266, 'larder': 5319, 'table': 4961, 'pinty': 269, 'generically': 270, 'honey': 271, 'indicates': 272, 'stiped': 2770, 'soars': 4015, 'aqua': 4935, 'want': 274, 'eyebar': 275, 'pipit': 517, 'woodpecker': 277, 'attract': 278, 'clad': 2453, 'flaring': 4367, 'scales': 4267, 'motion': 1846, 'ones': 4836, 'end': 281, 'nd': 5013, 'turn': 1847, 'provide': 283, 'excessively': 5045, 'lo9oking': 285, 'sitting': 2773, 'coloed': 287, 'feature': 288, 'speckelled': 290, 'how': 1715, 'amazing': 292, 'low': 2594, 'adjacent': 294, 'blackish': 295, 'cheetah': 296, 'widespread': 297, 'beach': 298, 'whitye': 2775, 'beack': 301, 'whitehead': 3701, 'pokes': 303, 'description': 1796, 'beauty': 305, 'fether': 54, 'ladder': 307, 'tethers': 308, 'winch': 5345, 'abdomen': 310, 'lump': 311, 'modest': 313, 'footed': 3077, 'lay': 315, 'spots': 5316, 'cement': 3733, 'law': 317, 'underpart': 318, 'parallel': 319, 'headdress': 320, 'shirt': 3704, 'colorfully': 323, 'greey': 324, 'splendid': 325, 'third': 326, 'plumaged': 327, 'greet': 331, 'encompassing': 332, 'checkerboard': 333, 'green': 334, 'specks': 335, 'africa': 2914, 'things': 336, 'greeb': 337, 'backwith': 524, 'lingala': 339, 'wing': 340, 'greed': 341, 'wine': 342, 'flippers': 343, 'spotting': 3609, 'withing': 4583, 'restrictiveness': 345, 'over': 346, 'orang': 348, 'sickle': 349, 'underside': 350, 'proportionate': 351, 'belill': 352, 'mallard': 354, 'rump': 4981, 'greyer': 3711, 'budding': 358, 'brownbreast': 1970, 'secondaries': 360, 'featerhs': 361, 'before': 362, 'beside': 363, 'accent': 4629, 'mottling': 364, 'plainly': 4605, 'frisky': 366, 'sunkin': 2492, 'shead': 368, 'highlights': 369, 'witrh': 370, 'avocado': 371, 'blackened': 3902, 'crew': 374, 'better': 375, 'tilts': 376, 'distinguishing': 377, 'fade': 378, 'distal': 1863, 'accentuating': 380, 'vents': 381, 'hidden': 382, 'blackeye': 384, 'strength': 3737, 'loght': 251, 'downy': 388, 'then': 389, 'them': 390, 'hgas': 391, 'shad': 2913, 'combination': 393, 'breas': 265, 'similar': 4007, 'slate': 395, 'wingtip': 5356, 'finish': 3187, 'breat': 397, 'break': 398, 'band': 399, 'lenth': 401, 'they': 403, 'sideways': 4558, 'length': 4592, 'comprised': 197, 'silver': 405, 'bank': 406, 'bread': 407, 'shadows': 408, 'bony': 410, 'outter': 1535, 'fringed': 412, 'rocky': 413, 'rectrices': 4616, 'regularly': 4821, 'fuscia': 415, 'reasonably': 2465, 'l': 417, 'rocks': 418, 'dingy': 419, 'noticable': 420, 'camoflauged': 421, 'adorns': 3364, 'arrow': 422, 'each': 423, 'feeds': 424, 'pinstripes': 3015, 'side': 426, 'mean': 427, 'superpilliary': 428, 'lifted': 429, 'hanger': 430, 'nave': 431, 'yelwo': 432, 'fairly': 433, 'cobalt': 4106, 'series': 434, 'mahogany': 435, 'whie': 2601, 'bornw': 437, 'tinting': 438, 'ligt': 439, 'contracting': 650, 'slabs': 4218, 'strips': 3727, 'irredentist': 442, 'vrown': 443, 'navy': 445, 'webbed': 446, 'whit': 447, 'recognizable': 448, 'potted': 1087, 'wingbared': 4134, 'borne': 451, 'nits': 452, 'velvet': 454, 'fevers': 455, 'gradient': 456, 'restricted': 457, 'ha': 4801, 'spekled': 458, 'hooked': 459, 'sand': 2802, 'gradation': 461, 'fellows': 462, 'rd': 585, 're': 464, 'thisis': 465, 'adapt': 466, 'got': 467, 'farhood': 468, 'feeat': 4633, 'swell': 470, 'interchanging': 471, 'splashed': 474, 'turning': 476, 'linear': 477, 'barrier': 478, 'colorless': 479, 'rail': 480, 'fave': 482, 'given': 1576, 'free': 484, 'beatty': 2868, 'proximally': 486, 'frey': 487, 'streamline': 488, 'splashes': 489, 'crwn': 490, 'crowned': 491, 'blandly': 386, 'skinny': 4768, 'bifid': 5344, 'alarmingly': 494, 'swimming': 5282, 'enormous': 495, 'ate': 496, 'birdie': 498, 'created': 499, 'iss': 500, 'starts': 501, 'appendages': 502, 'days': 504, 'descriptive': 5054, 'punctuated': 505, 'ish': 507, 'birdit': 508, 'isn': 509, 'incorporated': 510, 'signature': 511, 'punctuates': 512, 'moving': 3738, 'onto': 515, 'tufty': 472, 'rang': 518, 'bitd': 1256, 'angelic': 520, 'features': 521, 'grade': 522, 'detailing': 4269, 'inches': 3975, 'poofy': 525, 'primary': 526, 'thereof': 4225, 'hook': 528, 'featured': 529, 'downside': 531, 'another': 532, 'robust': 3742, 'scissors': 534, 'thick': 535, 'brakemen': 536, 'hood': 538, 'speck': 539, 'magnificent': 3856, 'koel': 541, 'similarly': 542, 'top': 543, 'struts': 544, 'perching': 545, 'twisted': 2272, 'approximately': 547, 'heights': 548, 'plentiful': 549, 'needed': 550, 'toi': 551, 'master': 552, 'too': 553, 'heas': 554, 'wildly': 555, 'mohogany': 556, 'inflate': 558, 'legs': 559, 'eats': 3073, 'ranging': 561, 'toe': 562, 'waiting': 4209, 'almond': 564, 'baclk': 565, 'fiery': 1898, 'compaired': 4676, 'outstandingly': 3262, 'centered': 567, 'distictive': 568, 'direct': 570, 'wrapped': 3297, 'stoic': 572, 'somewhat': 573, 'shortly': 998, 'brushed': 576, 'peculiar': 577, 'character': 578, 'begins': 579, 'distance': 580, 'additiont': 5303, 'disproportionally': 582, 'elegant': 584, 'windbars': 4565, 'bloated': 586, 'tree': 587, 'likely': 588, 'seated': 4, 'scant': 4635, 'project': 590, 'upset': 4758, 'silly': 592, 'aftereffects': 593, 'contrasted': 594, 'tipping': 595, 'minus': 596, 'bridge': 598, 'countershading': 600, 'bards': 601, 'underbody': 602, 'longs': 603, 'edgings': 2972, 'localized': 605, 'ram': 606, 'spectrum': 607, 'seed': 608, 'webfooted': 609, 'manner': 610, 'emphasizing': 611, 'sweeping': 4755, 'roufous': 612, 'seen': 1025, 'seem': 614, 'mint': 416, 'ibll': 616, 'relatively': 617, 'birght': 618, 'ragged': 1005, 'featherd': 2392, 'concrete': 1849, 'expressive': 623, 'latter': 624, 'retricies': 625, 'stomach': 4643, 'snow': 626, 'thorough': 627, 'predominantly': 628, 'oriented': 629, 'chest': 630, 'edge': 3758, 'quill': 632, 'gets': 5068, 'swims': 633, 'recitrices': 634, 'gradiants': 636, 'orangeish': 637, 'circles': 638, 'even': 639, 'retricles': 640, 'shallow': 4647, 'darkened': 3879, 'though': 642, 'what': 2838, 'shale': 644, 'anatomy': 645, 'y': 3760, 'involving': 647, 'mouth': 648, 'circled': 649, 'unorthodox': 652, 'glow': 654, 'known': 2970, 'peers': 656, 'whichecked': 657, 'don': 658, 'pointe': 659, 'nostrils': 661, 'speckeled': 662, 'backwards': 4580, 'duckling': 663, 'flow': 664, 'dog': 665, 'swamp': 666, 'points': 667, 'camo': 668, 'sun': 2844, 'pecker': 670, 'pointy': 671, 'splotching': 673, 'mating': 674, 'scream': 675, 'dot': 676, 'incorporate': 677, 'aquatic': 679, 'colord': 680, 'insects': 681, 'boack': 682, 'random': 683, 'sage': 684, 'ending': 685, 'pops': 686, 'colors': 688, 'showing': 5176, 'climb': 4257, 'dopey': 689, 'woodpecking': 690, 'subtle': 692, 'abreast': 694, 'marvelously': 695, 'hes': 4585, 'touches': 697, 'busy': 698, 'layout': 699, 'folded': 702, 'spekcles': 704, 'bush': 705, 'wingular': 5383, 'deepest': 1247, 'rich': 709, 'lodge': 710, 'greyt': 712, 'feeet': 713, 'wingbras': 714, 'wearing': 715, 'plate': 716, 'mixture': 717, 'colorful': 719, 'wide': 1928, 'dd': 721, 'de': 722, 'watch': 723, 'coast': 1164, 'bellly': 725, 'h': 4851, 'browinish': 727, 'geen': 728, 'intermingled': 730, 'vividly': 732, 'quarters': 3774, 'blacl': 4985, 'runs': 736, 'mainlly': 737, 'bar': 738, 'bas': 739, 'greens': 740, 'covering': 741, 'sautty': 742, 'fields': 743, 'skirts': 5363, 'itsbody': 2362, 'patch': 746, 'twice': 747, 'interspersed': 748, 'softly': 749, 'neckband': 4555, 'belyy': 1158, 'flanked': 752, 'eventually': 2467, 'eart': 754, 'complimenting': 1943, 'fluttered': 2600, 'steak': 756, 'steal': 757, 'bak': 758, 'ears': 759, 'bleu': 760, 'besk': 761, 'blew': 762, 'bea': 4668, 'splotched': 764, 'swept': 765, 'habit': 766, 'diminutive': 767, 'fluffy': 768, 'winbar': 769, 'yellowbill': 770, 'plumage': 773, 'splotches': 774, 'full': 2864, 'bed': 4670, 'barring': 3999, 'fail': 778, 'sporadic': 779, 'bee': 4671, 'nub': 781, 'yes': 1032, 'best': 783, 'subject': 784, 'pastel': 785, 'said': 786, 'wingles': 787, 'inappropriate': 788, 'lots': 789, 'away': 791, 'afloat': 792, 'rings': 793, 'siding': 1936, 'laarge': 795, 'propitiate': 796, 'luminescent': 797, 'orangge': 5136, 'finger': 798, 'shaggy': 2967, 'thisthis': 799, 'drawn': 800, 'sorts': 801, 'ird': 802, 'claws': 803, 'we': 804, 'terms': 805, 'wears': 806, 'wa': 1211, 'nature': 808, 'jetblack': 809, 'weak': 811, 'however': 812, 'wi': 813, 'hairlike': 814, 'stripped': 4017, 'nistly': 2211, 'wth': 816, 'scondaries': 817, 'flowering': 818, 'packages': 819, 'holds': 4392, 'deathers': 820, 'faced': 821, 'irregular': 822, 'met': 823, 'resistant': 824, 'ruffly': 2877, 'ill': 825, 'bronzed': 826, 'against': 827, 'edium': 828, 'feathering': 829, 'merging': 239, 'layered': 831, 'lively': 4734, 'mildly': 833, 'cherry': 834, 'distinction': 835, 'downsloping': 4683, 'tinge': 837, 'col': 838, 'peacefully': 5115, 'alternate': 3792, 'tone': 842, 'very': 2774, 'spear': 843, 'had': 844, 'boron': 846, 'redwith': 1947, 'height': 848, 'inquisitive': 56, 'matte': 851, 'gleaming': 852, 'stubbier': 372, 'blending': 854, 'majestic': 855, 'brindle': 856, 'humming': 857, 'starkwhite': 859, 'mannish': 860, 'yelklow': 861, 'speak': 862, 'brassy': 863, 'tint': 864, 'abdominal': 865, 'nvent': 866, 'hues': 3797, 'chrome': 868, 'beek': 869, 'chroma': 870, 'three': 871, 'been': 613, 'quickly': 873, 'beep': 874, 'fowl': 4688, 'tine': 876, 'spread': 1046, 'distinctie': 878, 'interest': 880, 'basic': 881, 'expected': 882, 'flexible': 883, 'lovely': 884, 'tallest': 885, 'wintgs': 5362, 'dots': 888, 'yellows': 3162, 'life': 890, 'deeper': 891, 'dominant': 4385, 'ebyll': 894, 'onyx': 895, 'gently': 4692, 'catch': 898, 'worked': 21, 'glides': 902, 'ascents': 903, 'colored': 2887, 'mediium': 51, 'exception': 906, 'has': 907, 'tank': 2245, 'auras': 909, 'reddish': 910, 'kingfisher': 912, 'somewhatlighter': 913, 'tand': 914, 'vbill': 4089, 'warbler': 915, 'air': 916, 'voluminous': 917, 'near': 918, 'abuts': 920, 'gape': 3246, 'stopping': 922, 'caretakers': 923, 'windbar': 924, 'freathers': 925, 'tans': 926, 'leaves': 927, 'characterizing': 928, 'mousy': 930, 'cylinder': 931, 'soild': 932, 'slicky': 933, 'undercarriage': 1704, 'is': 935, 'molarpatch': 936, 'it': 937, 'sticking': 2910, 'proportioned': 938, 'throat': 4701, 'cant': 941, 'cone': 943, 'exterior': 944, 'in': 267, 'ia': 946, 'otherside': 5304, 'mouse': 950, 'if': 951, 'grown': 952, 'containing': 953, 'shimmering': 954, 'bilious': 955, 'undertailed': 956, 'wingabrs': 957, 'make': 958, 'protrudes': 4980, 'donning': 1688, 'who': 4736, 'biege': 4124, 'forehead': 962, 'compact': 4115, 'complex': 963, 'upon': 5080, 'belly': 964, 'mixtures': 396, 'bellt': 966, 'several': 968, 'grows': 1750, 'european': 971, 'wheel': 972, 'meets': 973, 'bearded': 974, 'winge': 3814, 'cheeks': 4306, 'peaks': 976, 'tanager': 3763, 'kis': 978, 'alittle': 979, 'pinstripe': 475, 'tops': 981, 'differing': 485, 'opulent': 983, 'hand': 984, 'tanish': 985, 'elly': 986, 'astonishingly': 1973, 'outmost': 988, 'scarlet': 4024, 'workings': 989, 'tope': 990, 'albino': 3817, 'overs': 992, 'downwardly': 993, 'tune': 994, 'midsized': 995, 'butter': 996, 'windy': 4109, 'wispy': 997, 'kept': 575, 'smokey': 999, 'brouwn': 2745, 'wisps': 1001, 'ths': 1162, 'thr': 1003, 'ocean': 1004, 'eyebrew': 619, 'glimmers': 1976, 'hearty': 1007, 'qualities': 1008, 'descending': 1009, 'the': 1010, 'smoked': 1011, 'possesses': 1012, 'wider': 5209, 'left': 1013, 'ismostly': 503, 'bellow': 1015, 'indigo': 1016, 'background': 1017, 'just': 1018, 'golden': 174, 'athletic': 1020, 'photo': 1021, 'sporting': 1022, 'greys': 1023, 'sheeny': 1024, 'mid': 4717, 'rodent': 735, 'body': 4220, 'identify': 1027, 'thanks': 238, 'human': 1029, 'partridge': 621, 'upturned': 1031, 'discoloration': 707, 'yet': 1034, 'snout': 1980, 'depicts': 1036, 'sunflower': 1037, 'furry': 1038, 'noticible': 1039, 'royal': 1040, 'patching': 1042, 'hilly': 858, 'chartreuse': 1045, 'n': 98, 'spectacled': 1047, 'board': 1048, 'easy': 1049, 'jeweled': 1050, 'offwhite': 1051, 'save': 1052, 'hat': 1053, 'comouflaged': 1054, 'trimming': 1056, 'pluage': 1057, 'turquoise': 1058, 'boxed': 1059, 'roosting': 1060, 'bands': 1062, 'feets': 1063, 'possible': 1066, 'spanned': 1067, 'possibly': 1068, 'arebrown': 1069, 'discreet': 1070, 'wiht': 1071, 'birth': 1072, 'clustered': 1073, 'cape': 1074, 'shadow': 1075, 'unique': 1076, 'burnt': 1077, 'bushy': 1079, 'apart': 1080, 'shoulder': 1081, 'foots': 1082, 'articulated': 1083, 'appearing': 1084, 'belloed': 703, 'tapering': 1086, 'shaped': 3831, 'thunderbolt': 3190, 'agray': 1088, 'flecked': 5372, 'fluffly': 1090, 'sparse': 1091, 'night': 1092, 'throught': 4541, 'triangularbill': 1095, 'soar': 4729, 'interestingly': 3833, 'overtone': 1098, 'flappy': 3830, 'webb': 1099, 'right': 1101, 'old': 1102, 'creek': 1103, 'crowd': 1104, 'flatter': 1106, 'overhung': 1107, 'mohock': 190, 'crown': 1109, 'begin': 1992, 'proportionally': 1111, 'eyeing': 1112, 'born': 1113, 'stutbby': 5152, 'clawed': 1114, 'spoted': 1115, 'winds': 4112, 'borg': 1116, 'unless': 4731, 'chesty': 1093, 'bord': 1118, 'ambered': 1120, 'nevk': 1311, 'knifelike': 1122, 'festers': 1123, 'brd': 1124, 'dense': 1125, 'ruffled': 1126, 'for': 1127, 'bottom': 1128, 'purple': 1129, 'dropping': 3627, 'normal': 1997, 'hodge': 1132, 'poupon': 1133, 'ice': 1134, 'iridescently': 1135, 'everything': 1136, 'ruffles': 1137, 'cird': 1138, 'eyebrowed': 1139, 'freshers': 1140, 'fom': 1141, 'beating': 1142, 'adorable': 1143, 'coller': 1146, 'particular': 2969, 'billwith': 1148, 'bold': 1149, 'fish': 3181, 'corn': 1151, 'burn': 1152, 'graduates': 1153, 'narow': 1154, 'ple': 1155, 'secondsries': 1156, 'shifting': 1157, 'graduated': 1159, 'post': 1160, 'super': 1161, 'outers': 4235, 'transcends': 1163, 'yelloiw': 4855, 'ored': 1166, 'abdomin': 1167, 'plum': 1168, 'rim': 3693, 'surround': 1206, 'sequin': 3845, 'exeception': 1171, 'distinct': 2932, 'flourishes': 1173, 'midlength': 1174, 'plus': 1175, 'willows': 3997, 'done': 3271, 'steadily': 1177, 'azure': 1178, 'connected': 4741, 'seabird': 1180, 'slightly': 1181, 'boldy': 23, 'beinga': 1183, 'raised': 1185, 'thinner': 1700, 'agile': 1187, 'coverlets': 1188, 'mantle': 1189, 'ducking': 2527, 'float': 1190, 'bound': 1192, 'son': 1562, 'down': 1194, 'ahs': 1761, 'trimmed': 5220, 'wingsand': 1196, 'consumption': 4289, 'capped': 1197, 'coastal': 1198, 'pluffed': 4745, 'balanced': 1199, 'wingback': 3849, 'shoots': 1201, 'eblly': 1799, 'rounds': 1807, 'grrenish': 1811, 'strangely': 1205, 'grebe': 4554, 'unicolor': 1207, 'lashes': 1208, 'weathered': 1209, 'amazingly': 1210, 'flying': 1212, 'fuchsia': 1213, 'sunlight': 1214, 'littering': 2010, 'width': 1217, 'grasping': 1219, 'way': 1220, 'lage': 1222, 'bobolink': 1223, 'blurry': 5236, 'fork': 1225, 'greyish': 2012, 'head': 1227, 'faring': 1228, 'heac': 1229, 'form': 1230, 'snowy': 1231, 'something': 4377, 'forming': 1984, 'becoming': 1233, 'ofver': 1234, 'varigataed': 1235, 'vovert': 4751, 'landing': 1237, 'back': 3857, 'mysterious': 2945, 'heat': 1239, 'peice': 2909, 'hear': 1241, 'dead': 1242, 'ununiformally': 1243, 'beady': 4513, 'reticles': 2059, 'true': 1246, 'mediup': 718, 'cheeck': 1249, 'mouthed': 1250, 'trinagular': 2947, 'portions': 1253, 'flecks': 1254, 'featrues': 1255, 'inside': 1257, 'largish': 2017, 'attached': 1259, 'tell': 1261, 'follows': 3832, 'streamlined': 4859, 'decorate': 1263, 'sapphire': 1264, 'coverings': 1265, 'smashed': 1266, 'unusually': 1267, 'whisker': 1268, 'more': 2951, 'rectricle': 1270, 'adorn': 1272, 'originating': 1273, 'brightly': 1274, 'juvenile': 1275, 'sticks': 1276, 'winged': 3094, 'inclusive': 1277, 'fury': 1278, 'covert': 1279, 'abstract': 1280, 'intricately': 2800, 'birdhas': 1282, 'covers': 1283, 'eings': 1284, 'meanwhile': 3761, 'aboce': 5335, 'contour': 3537, 'baige': 3866, 'splotch': 1288, 'sprouts': 1289, 'floating': 1290, 'whiskered': 1291, 'check': 1293, 'glank': 1294, 'reticules': 1295, 'ciliaries': 1296, 'marking': 1297, 'outstretched': 1298, 'dotted': 1299, 'no': 1300, 'bedhead': 1301, 'na': 1302, 'whereas': 1303, 'generally': 1304, 'topped': 1305, 'eyeings': 1306, 'ablue': 2959, 'tin': 1309, 'setting': 1310, 'holding': 1312, 'digital': 1313, 'test': 1314, 'tie': 1315, 'orchid': 1316, 'crowna': 1317, 'crownb': 1318, 'greenery': 1319, 'rainbird': 1320, 'aroung': 5394, 'toad': 1321, 'felt': 1322, 'uniformly': 1323, 'gradually': 3870, 'taurus': 1325, 'brief': 2848, 'thorat': 1328, 'scale': 3871, 'heded': 1329, 'smattered': 1330, 'dappled': 1331, 'transfers': 959, 'scratchy': 1333, 'existent': 4256, 'thorax': 1335, 'progresses': 1336, 'longer': 1337, 'bullet': 1338, 'spects': 1339, 'pefect': 2637, 'cuerved': 1341, 'together': 1342, 'collared': 1343, 'rough': 4794, 'wingand': 1344, 'fuzze': 4644, 'time': 1346, 'serious': 1347, 'backward': 1348, 'dust': 5086, 'songs': 1349, 'ringbars': 1350, 'peacock': 1351, 'yekkiw': 1352, 'plumeage': 1353, 'remarkable': 1354, 'dance': 1355, 'fox': 1356, 'flst': 2657, 'walking': 4889, 'neckring': 1359, 'leads': 1360, 'icolored': 2794, 'mild': 1362, 'skim': 1364, 'astriped': 1366, 'skin': 1367, 'shot': 2039, 'remarkably': 1369, 'plucked': 1370, 'displaying': 1371, 'midst': 1372, 'row': 1373, 'layers': 1374, 'vet': 1375, 'inverse': 1376, 'ver': 1377, 'tufted': 2040, 'hovers': 1379, 'birdwith': 1380, 'hump': 1381, 'flash': 1382, 'father': 1383, 'smoothly': 5203, 'vey': 939, 'environment': 1385, 'finally': 1386, 'vee': 1387, 'swoop': 1388, 'ovular': 1389, 'sing': 4576, 'hwit': 1391, 'eybrows': 1393, 'brown': 1394, 'curvage': 1395, 'protective': 1396, 'string': 1397, 'browh': 948, 'happens': 5216, 'bired': 1399, 'outerreachers': 1400, 'browb': 1401, 'seemingly': 1402, 'sloppy': 1403, 'cutting': 4491, 'tectrices': 1405, 'curve': 5378, 'shrot': 46, 'stays': 1407, 'cyan': 1408, 'emphasized': 1409, 'little': 1884, 'brows': 1410, 'duller': 3171, 'cool': 1412, 'skyblue': 1413, 'flittery': 1416, 'impressive': 1417, 'level': 1418, 'tear': 1419, 'turns': 1420, 'fragile': 1421, 'brother': 1422, 'charcoal': 4308, 'pea': 3885, 'realtionship': 1424, 'lightpink': 1425, 'brownish': 1426, 'q': 4708, 'teal': 1427, 'gut': 1428, 'quick': 1429, 'spiral': 5315, 'guy': 1430, 'woven': 1432, 'pompadour': 1404, 'upper': 1434, 'tailtip': 3420, 'tremendously': 949, 'blackand': 1436, 'meadow': 1437, 'core': 316, 'muscularly': 1439, 'eyerung': 1440, 'rouge': 4793, 'specked': 1442, 'steaks': 1443, 'chead': 2934, 'hsa': 1445, 'crescent': 1446, 'corner': 2053, 'run': 2853, 'wingbar': 1450, 'winbgbars': 1451, 'bakc': 1452, 'turquorise': 1453, 'ches': 1454, 'bdody': 1455, 'stands': 1456, 'feat': 4796, 'blotches': 1457, 'coppery': 1458, 'uniform': 1460, 'reminiscent': 1461, 'hwite': 3890, 'beard': 4639, 'lazily': 1463, 'goes': 1464, 'ventrally': 5239, 'falling': 1466, 'order': 2024, 'ivory': 5330, 'blotched': 1468, 'alertness': 1469, 'hairy': 1470, 'supporting': 1471, 'foliage': 1473, 'dweller': 1474, 'nearer': 4799, 'heavily': 3453, 'blunted': 1477, 'hairs': 1478, 'mnostly': 1479, 'tarsis': 4800, 'floats': 4984, 'perpendicular': 2686, 'water': 1483, 'evident': 5352, 'snack': 5030, 'witch': 1486, 'lustrous': 1487, 'spattering': 1488, 'alone': 3607, 'along': 1490, 'mosly': 1491, 'rubbery': 1492, 'appears': 1493, 'ombres': 1494, 'downturned': 5164, 'boy': 1495, 'pearly': 1496, 'standout': 1497, 'cuckoo': 889, 'brilliant': 37, 'shift': 1500, 'dominantly': 1501, 'blotch': 1502, 'vibrating': 1503, 'widening': 4241, 'bow': 1504, 'smnall': 1505, 'throughput': 1506, 'cinnamon': 1507, 'usually': 1078, 'weird': 1510, 'supercilious': 2170, 'bod': 1512, 'pattern': 4892, 'stubbed': 1513, 'sleak': 1516, 'love': 1517, 'expects': 4884, 'shelled': 1518, 'extra': 1519, 'gull': 2066, 'flanks': 3901, 'compare': 3004, 'marked': 1523, 'cloak': 2067, 'marvelous': 1525, 'retires': 1526, 'puffed': 1527, 'shimmer': 2068, 'tailand': 1529, 'fake': 218, 'blck': 1531, 'forefront': 1532, 'smallish': 1533, 'whiet': 1534, 'sported': 1536, 'are': 3530, 'identically': 5122, 'mocha': 4774, 'upperside': 1538, 'vivid': 1539, 'sharper': 1540, 'brighter': 4812, 'dalmatian': 1542, 'peri': 967, 'contoured': 1545, 'only': 2072, 'tightly': 1547, 'sports': 1548, 'sharped': 1549, 'creamy': 1550, 'going': 2073, 'prey': 1552, 'opposed': 1553, 'pert': 1554, 'earthly': 5287, 'torund': 4813, 'blonde': 5215, 'thru': 1556, 'chechpatch': 1557, 'ands': 1558, 'loving': 1559, 'blotchy': 1560, 'shine': 3908, 'smudged': 4020, 'pattered': 1563, 'variegation': 1564, 'malar': 1565, 'redbill': 1566, 'retrices': 1567, 'apparent': 1568, 'chocolate': 1569, 'marbled': 1570, 'drown': 1571, 'everywhere': 1572, 'flapping': 1573, 'olive': 1574, 'shafts': 3030, 'featureless': 3952, 'fly': 1579, 'wngspan': 3712, 'bleeds': 3922, 'breaston': 1582, 'car': 1584, 'towhee': 1585, 'cap': 1586, 'caw': 1587, 'cat': 1589, 'greying': 3913, 'protrude': 4190, 'cao': 1592, 'can': 1593, 'recticles': 1595, 'following': 1596, 'frilled': 1597, 'midium': 1599, 'bites': 4246, 'coral': 1601, 'streak': 1602, 'heart': 1603, 'crazy': 1604, 'grassy': 1605, 'figure': 1606, 'expanding': 1607, 'awesome': 1608, 'shimmers': 1609, 'heard': 1612, 'tinted': 1613, 'chin': 1614, 'o': 1170, 'spn': 1616, 'brest': 276, 'streas': 1618, 'spanning': 1619, 'distinguishable': 1620, 'airplane': 2277, 'magenta': 3020, 'hatchling': 1622, 'pink': 1623, 'winter': 1624, 'tilt': 1627, 'divided': 1628, 'spreads': 1176, 'poking': 1630, 'retices': 1631, 'species': 3021, 'strikingly': 1633, 'till': 1634, 'distally': 1635, 'flank': 1636, 'cardinal': 1637, 'aare': 5326, 'pure': 1638, 'haveing': 1640, 'gaping': 1641, 'speaks': 1642, 'chirpy': 4110, 'pint': 1643, 'rivaled': 1644, 'divi': 1645, 'may': 1646, 'neat': 1647, 'use': 4285, 'stouts': 560, 'marbling': 1650, 'vthis': 4722, 'tarsal': 4595, 'misshapen': 1652, 'tipped': 1653, 'speckling': 1654, 'gradiently': 1655, 'tarus': 1656, 'such': 1657, 'yeallow': 1658, 'dove': 1659, 'shockingly': 1660, 'hinds': 4976, 'man': 1662, 'slashes': 1908, 'climbing': 3273, 'branches': 1358, 'neck': 1666, 'beam': 4834, 'maybe': 1669, 'brownm': 3026, 'yielding': 1671, 'daisies': 1672, 'tale': 1673, 'rather': 3027, 'switch': 1462, 'sh': 1676, 'truck': 1677, 'brigtly': 1678, 'swollen': 1679, 'thorns': 1680, 'tall': 1681, 'croon': 4185, 'ores': 1682, 'se': 1683, 'typical': 1684, 'cute': 1685, 'tagged': 1686, 'breaking': 3029, 'birdh': 314, 'birdi': 1689, 'freckles': 1691, 'pointed': 1692, 'predominently': 1693, 'displayed': 1145, 'primitive': 1695, 'canary': 1696, 'outer': 4459, 'decently': 3244, 'ended': 1699, 'corverts': 4075, 'goldfinch': 1701, 'pointex': 1702, 'pitch': 1703, 'cold': 1705, 'still': 1706, 'owl': 5284, 'birds': 1707, 'freckled': 1708, 'blocked': 1709, 'solitary': 1710, 'ovals': 58, 'group': 1712, 'feath': 1713, 'curly': 1714, 'interesting': 1716, 'plethora': 3929, 'curls': 1719, 'hot': 1720, 'forms': 1722, 'window': 1723, 'offers': 1724, 'spacious': 1725, 'coordinated': 1726, 'main': 1727, 'parades': 1728, 'eyecatching': 4599, 'beid': 1730, 'modestly': 1731, 'non': 1732, 'completly': 1733, 'sueperciliary': 1734, 'tucked': 1735, 'weathers': 1736, 'matches': 1737, 'tarsas': 1739, 'taildfeather': 899, 'rectices': 1741, 'chromatic': 1742, 'underneath': 1743, 'yields': 3839, 'touching': 1745, 'mutliple': 1216, 'covarts': 1747, 'half': 1748, 'not': 1749, 'thus': 5040, 'now': 1611, 'patter': 1752, 'ordinary': 1753, 'matched': 1755, 'tern': 1756, 'grye': 205, 'streaks': 1760, 'corners': 1762, 'drop': 1763, 'arrows': 1764, 'silverbill': 3937, 'nack': 4695, 'recrices': 1768, 'finely': 1769, 'wigs': 3040, 'rock': 1771, 'dabs': 1365, 'entirely': 1773, 'quarter': 1651, 'backish': 1776, 'yead': 1777, 'square': 1778, 'ruffle': 2865, 'eh': 162, 'reeds': 4363, 'ed': 240, 'growths': 1781, 'directing': 1782, 'snapshots': 1783, 'goose': 1784, 'indentified': 3941, 'gird': 1786, 'catching': 1787, 'ey': 1788, 'zebra': 1789, 'preaching': 1790, 'ablack': 1791, 'frilly': 3045, 'spiking': 1794, 'distended': 1795, 'after': 5256, 'oatmeal': 1797, 'shown': 1798, 'opened': 1800, 'space': 1801, 'blu': 1802, 'beedy': 3928, 'furthermore': 1804, 'perwinkle': 7, 'looking': 1808, 'bll': 1809, 'beige': 4407, 'spangled': 3533, 'florescent': 3048, 'waterfowl': 3080, 'sorrounding': 66, 'coverimg': 1816, 'bordering': 1144, 'superiors': 84, 'smalls': 1819, 'complimented': 1820, 'dorsally': 1821, 'monster': 1822, 'tallons': 1823, 'hefty': 1824, 'unusal': 1825, 'fuschia': 1826, 'quite': 1827, 'muddied': 5286, 'smallb': 1828, 'smalle': 1829, 'rebel': 172, 'smallg': 177, 'bumpy': 1832, 'ominous': 1833, 'marine': 200, 'smalll': 1835, 'ora': 1836, 'reflect': 3054, 'gunmetal': 1837, 'silhouetted': 1838, 'disguised': 1839, 'feathersm': 247, 'flare': 1842, 'transition': 1843, 'punk': 1844, 'peppering': 1845, 'mohalk': 279, 'thing': 282, 'funky': 1848, 'plane': 3377, 'place': 1850, 'massive': 1851, 'superciiliary': 1852, 'swing': 1853, 'plack': 790, 'view': 4866, 'think': 1856, 'first': 1392, 'origin': 234, 'pelican': 344, 'supercillaries': 1860, 'dwelling': 1861, 'striking': 1862, 'zag': 1105, 'coming': 3060, 'stribed': 1865, 'ong': 1866, 'withwhite': 1867, 'one': 404, 'soaring': 1869, 'gelly': 1870, 'long': 3959, 'directly': 1872, 'shadowed': 1873, 'comprises': 373, 'chubby': 1875, 'ring': 449, 'open': 1878, 'angular': 1879, 'ont': 1880, 'size': 1881, 'squashed': 1882, 'interchange': 1883, 'turouise': 483, 'checked': 1885, 'stuck': 4600, 'whitw': 1888, 'murky': 1889, 'bite': 1890, 'breed': 519, 'whitr': 1893, 'anteriorly': 1894, 'whos': 3624, 'checker': 1895, '2': 1896, 'bleak': 3064, 'grackles': 566, 'whith': 1899, 'spectacularly': 3059, 'spackling': 1901, 'white': 1902, 'sheek': 1903, 'bits': 1904, 'frame': 4870, 'sheen': 1906, 'gives': 1907, 'multicored': 655, 'hue': 1909, 'godied': 359, 'abird': 1911, 'mostly': 1912, 'that': 1913, 'streaked': 1914, 'sicks': 3755, 'short': 3068, 'eyee': 1920, 'eyed': 1921, 'porcupine': 3465, 'butt': 1922, 'occurring': 1924, 'flowers': 1925, 'than': 1926, '11': 1927, '10': 720, '13': 1929, '12': 1930, 'rugged': 1931, '14': 1932, 'earthtoened': 3214, '18': 753, 'victorian': 794, 'glossy': 1937, 'were': 1938, 'gobble': 1940, 'gigantic': 1941, 'almos': 1942, 'notched': 1944, 'powder': 4877, 'and': 1946, 'nad': 4038, 'anf': 847, 'camouflaged': 1948, 'puddly': 309, 'featehrs': 1950, 'considerably': 3970, 'sac': 1952, 'stubbu': 1953, 'ann': 1954, 'graduating': 4195, 'turned': 1955, 'locations': 1956, 'generalization': 1957, 'areyellow': 901, 'dash': 905, 'winding': 1960, 'slam': 919, 'camouflages': 1962, 'larvae': 5309, 'ans': 1963, 'flatness': 1964, 'monochrome': 1965, 'slab': 3708, 'any': 1968, 'retifices': 1971, 'lumpy': 982, 'hooded': 987, 'wnd': 1974, 'coal': 1975, 'efficient': 1006, 'tri': 4154, 'downwards': 1977, 'aside': 1978, 'zoo': 1979, 'note': 1035, 'other': 5407, 'abdomend': 1981, 'squad': 1982, 'take': 1983, 'resembling': 1985, 'upside': 1986, 'primaries': 1987, 'blunt': 1988, 'heal': 329, 'butterfly': 1990, 'inky': 1991, 'topping': 1110, 'sure': 1993, 'multiple': 1994, 'shade': 3081, 'pail': 1996, 'opposite': 1131, 'darkish': 1061, 'falls': 2001, 'talk': 4764, 'longtailed': 1165, 'sheened': 4818, 'fce': 2003, 'billow': 2004, 'pair': 1195, 'knee': 2006, 'coveted': 2008, 'leapord': 2009, 'bluebird': 1215, 'thsi': 1226, 'retrice': 2013, 'pathes': 1240, 'especially': 2015, 'dragonfly': 2016, 'fills': 1258, 'encased': 2019, 'average': 2020, 'proud': 2021, 'hungry': 2022, 'gracefully': 2023, 'promaries': 469, 'beaded': 2025, 'virtually': 3715, 'sunset': 2026, 'backdrop': 1332, 'obsidian': 2028, 'tartus': 2029, 'dispersed': 2030, 'dwhite': 2031, 'salt': 1340, 'accented': 2033, 'erred': 893, 'perch': 2036, 'shor': 2037, 'wingspann': 4680, 'spaced': 2011, 'eyesand': 1378, 'show': 2041, 'cheat': 2042, 'able': 4716, 'cheap': 2044, 'adorned': 3089, 'vreass': 2046, 'andblack': 2048, 'ling': 3982, 'thcik': 599, 'cheak': 1441, 'flack': 1444, 'scarce': 1447, 'aggressive': 2054, 'eyelids': 4549, 'ground': 1467, 'enthusiast': 2056, 'protrusion': 2057, 'sinks': 2058, 'slow': 4172, 'ratio': 2060, 'brigh': 2061, 'stormy': 2062, 'mohawked': 3985, 'plume': 2064, 'stained': 2065, 'proportion': 1520, 'plumb': 1524, 'texture': 1528, 'considerable': 2314, 'blacn': 2071, 'blacm': 1546, 'wood': 1551, 'black': 2074, 'berk': 1248, 'equipped': 2076, 'uppermost': 2077, 'stain': 2078, 'winglines': 1581, 'wingers': 2080, 'pumpkin': 132, 'lighted': 2083, 'swooped': 3844, 'narrows': 2085, 'get': 2087, 'subcilliary': 2088, 'vary': 2089, 'gey': 2090, 'truly': 2091, 'cannot': 2093, 'nearly': 2094, 'lighter': 2095, 'distinctly': 2096, 'symmetrically': 2097, 'fluffiness': 2098, 'intermixed': 2099, 'blends': 2100, 'secondary': 2101, 'tallon': 2102, 'swim': 4943, 'whute': 2052, 'held': 3101, 'pouched': 2108, 'median': 1780, 'hibernation': 2110, 'eyring': 2111, 'across': 2160, 'naked': 2113, 'undertone': 2115, 'bisects': 2161, 'characteristic': 3995, 'sutle': 2118, 'asnd': 2119, 'wiry': 4909, 'where': 2121, 'rims': 4894, 'poibty': 4988, 'tortoise': 2981, 'ballooned': 1108, 'burst': 2126, 'imposing': 2127, 'eyebro': 2128, 'ends': 4911, 'fanned': 2130, 'enck': 1892, 'relative': 2132, 'yellowbelly': 2134, 'us': 2135, 'flatly': 2136, 'seal': 2137, 'thoat': 2138, 'sport': 2140, 'torsus': 2141, 'slimmer': 2142, 'ability': 4914, 'appear': 3407, 'palette': 2144, 'ways': 1877, 'alarming': 2147, 'opening': 4916, 'blcak': 2148, 'isblack': 2149, 'monotone': 3224, 'teeny': 2150, 'limbed': 2151, 'pump': 2051, 'breatss': 2153, 'complexion': 2154, 'winger': 3112, 'between': 2156, 'treebird': 2158, 'enjoys': 2159, 'checks': 2112, 'oversized': 2116, 'eyerings': 2163, 'notice': 2164, 'checkered': 2165, 'vertical': 2166, 'crey': 2167, 'faded': 4002, 'dome': 2171, 'blame': 2172, 'adept': 2173, 'bblue': 2174, 'ourter': 2175, 'menacing': 2176, 'rapidly': 3115, 'spark': 2178, 'comb': 2179, 'come': 2180, 'concentrated': 2182, 'circumference': 2183, 'discernible': 2184, 'fit': 2185, 'irredecent': 2186, 'many': 2188, 'region': 2189, 'clown': 5183, 'quiet': 2190, 'contract': 2191, 'called': 4008, 's': 2193, 'berry': 2195, 'smallbrown': 2196, 'multi': 5006, 'slight': 5241, 'expression': 2198, 'comes': 2199, 'nearby': 2200, 'spand': 2201, 'vibrantly': 2202, 'secondaris': 2203, 'key': 4922, 'color': 2206, 'hunched': 2209, 'pop': 2210, 'talon': 4273, 'eyecircles': 2212, 'twin': 2213, 'pom': 2214, 'stretched': 2215, 'spans': 2216, 'butte': 2217, 'unassuming': 2218, 'bizarre': 2220, 'poll': 1411, 'polk': 2222, 'splotchy': 4698, 'poli': 2223, 'tiniest': 4014, 'considering': 2225, 'unusual': 2226, 'backand': 2227, 'tuning': 2229, 'capable': 2230, 'west': 2231, 'breats': 2123, 'mark': 2234, 'tropical': 2235, 'dashes': 4055, 'combined': 2238, 'needlelike': 2239, 'triangled': 2240, 'reflective': 2241, 'featuring': 2242, 'staring': 4928, 'hardly': 2243, 'melar': 2244, 'peaking': 2246, 'variagated': 1150, 'through': 3104, 'winks': 2805, 'juts': 2249, 'molt': 2250, 'elvis': 2251, 'formed': 2252, 'grayscale': 1292, 'hits': 4929, 'homely': 2254, 'uncurved': 2255, 'tiger': 2256, 'dramatic': 2258, 'muscular': 2260, 'variatied': 2261, 'multiclored': 3129, 'andbrown': 2263, 'attentive': 2265, 'ruby': 2266, 'diffuse': 2267, 'those': 2268, 'fluffing': 4752, 'eis': 2269, 'prehistoric': 2270, 'predominately': 2271, 'lye': 3042, 'these': 2274, 'fin': 2275, 'twigs': 2815, 'culverts': 3898, 'cast': 2278, 'diverse': 4026, 'slippery': 2280, 'hughed': 2281, 'mono': 2282, 'mound': 2283, 'hunts': 2284, 'necks': 4027, 'versus': 2287, 'vest': 2288, 'greyand': 2289, 'orientation': 2290, 'coupled': 2291, 'webbe': 2092, 'brow': 2293, 'strait': 2294, 'coffee': 2295, 'pinky': 2296, 'middle': 2297, 'biar': 1121, 'tiny': 872, 'converts': 2300, 'absomen': 2301, 'bgrayis': 2302, 'breastalong': 908, 'pinks': 2304, 'ink': 5386, 'shines': 2305, 'lightening': 4661, 'candied': 2307, 'compatriots': 2308, 'helmet': 2309, 'different': 2310, 'wintry': 2311, 'poited': 2312, 'wingspans': 2313, 'stomached': 2315, 'eggshell': 2316, 'spiraling': 2317, 'multicolor': 2318, 'same': 2319, 'anbd': 2320, 'reddiish': 394, 'wingspand': 2322, 'pae': 2323, 'scooped': 4104, 'lite': 2324, 'hite': 4939, 'steeper': 4033, 'contouring': 4617, 'narry': 2329, 'distinctively': 5028, 'mane': 2330, 'cheekpatches': 2331, 'noble': 2332, 'barbed': 2333, 'oil': 651, 'edged': 2335, 'wand': 2259, 'leaf': 5218, 'hunter': 3143, 'nest': 2339, 'bac': 4561, 'loarge': 2342, 'thickb': 4923, 'inbetween': 849, 'edges': 2344, 'delicate': 2345, 'changing': 2347, 'cheery': 2348, 'totally': 2349, 'thrown': 3239, 'eith': 2350, 'enchanting': 1390, 'mohawk': 2352, 'spotted': 4867, 'flaked': 2353, 'so': 4727, 'undefined': 687, 'largely': 2355, 'patchwork': 2356, 'bodied': 2357, 'mallor': 2358, 'eerie': 2359, 'cill': 4293, 'roughly': 2360, 'vback': 2361, 'stunted': 2363, 'severe': 2364, 'without': 2365, 'flakes': 2366, 'components': 2367, 'pewter': 2368, 'coordinate': 1414, 'broan': 4071, 'chrest': 2370, 'bodies': 2372, 'swatched': 2373, 'alfalfa': 2374, 'among': 2375, 'being': 2376, 'tip': 440, 'emerald': 4495, 'cranial': 4931, 'neon': 2379, 'stretches': 2380, 'rest': 2381, 'undertail': 2383, 'iwht': 2384, 'sizd': 2385, 'aspect': 845, 'touch': 2387, 'flavor': 2389, 'tis': 2390, 'speed': 1415, 'trimmings': 691, 'corvets': 1740, 'segal': 2395, 'pinkish': 2396, 'dipped': 2397, 'widest': 2399, 'hint': 2400, 'rigid': 4093, 'rose': 2401, 'seems': 2402, 'except': 2403, 'mirrored': 492, 'plumed': 2405, 'pallet': 2406, 'beye': 2407, 'lets': 2408, 'pile': 2409, 'pinkness': 2410, 'prettily': 2035, 'dully': 2412, 'plumes': 2413, 'crookedly': 4322, 'extensive': 2414, 'spackled': 2415, 'plimage': 2416, 'pill': 2417, 'disproportionate': 2418, 'seams': 227, 'boast': 2421, 'dary': 2422, 'hover': 2423, 'aspects': 2424, 'around': 2425, 'spectacular': 2426, 'read': 2427, 'outermost': 2428, 'specimen': 2429, 'mow': 2430, 'amd': 2431, 'dark': 2432, 'patterning': 4053, 'gnarled': 2434, 'throasr': 2435, 'riffles': 4956, 'pounted': 2437, 'darl': 2438, 'whte': 2439, 'world': 2440, 'swatch': 2441, 'barkbrown': 2442, 'superciliarys': 2443, 'supercilium': 2444, 'rear': 2445, 'meat': 2446, 'abrown': 2447, 'exeption': 3158, 'througout': 2450, 'identifying': 2451, 'earthy': 1891, 'abright': 1271, 'bracelet': 2455, 'pole': 2456, 'patterend': 734, 'lightning': 2458, 'facing': 2459, 'thighs': 1384, 'clay': 2461, 'completely': 4571, 'claw': 2462, 'downward': 2463, 'inter': 2464, 'grouping': 2276, 'sharp': 3019, 'brite': 2468, 'tending': 4918, 'on': 4313, 'grub': 2469, 'picture': 2470, 'creeper': 2471, 'puffin': 830, 'beaks': 5255, 'wattle': 2473, 'tannish': 2474, 'pacific': 2476, 'retrace': 4041, 'sback': 2478, 'diving': 2480, 'curvedbeak': 2481, 'intact': 2482, 'matching': 2483, 'naturally': 5328, 'alternates': 2484, 'ekes': 3764, 'twig': 2221, 'hoooked': 116, 'dimensions': 2489, 'yelloow': 2490, 'cavity': 2491, 'domed': 1855, 'tube': 1625, 'whaite': 2493, 'extensively': 473, 'stops': 2495, 'moon': 2496, 'flourescent': 2497, 'ellie': 2498, 'fringes': 2499, 'variated': 2500, 'featherless': 2501, 'provides': 2502, 'greyscale': 2503, 'stopped': 3165, 'forks': 2505, 'moderate': 2506, 'decent': 4971, 'reticule': 2507, 'turkey': 2508, 'beka': 2509, 'wheat': 2510, 'prominent': 5174, 'retains': 2513, 'nails': 2514, 'throa': 2515, 'notable': 2516, 'combines': 2454, 'premaries': 2518, 'notably': 2519, 'broken': 2520, 'diet': 2795, 'wingbands': 2522, 'found': 2523, 'contrasging': 4228, 'tarsals': 4412, 'tailfeather': 2526, 'throw': 1815, 'wader': 2528, 'comparison': 2529, 'stone': 2530, 'subdued': 2531, 'central': 2532, 'ace': 2533, 'blackcheek': 2534, 'ack': 2535, 'of': 2536, 'fringe': 2537, 'favorite': 2538, 'slender': 2539, 'mixes': 2540, 'veins': 5248, 'midnight': 4826, 'boldly': 2542, 'stand': 1245, 'oy': 2544, 'fringy': 2545, 'mixed': 2546, 'cheekstripe': 2547, 'ot': 2548, 'os': 2549, 'or': 2550, 'road': 2551, 'amber': 2552, 'clamp': 3173, 'whitey': 2554, 'lands': 2555, 'dagger': 2556, 'shapely': 2557, 'largecompared': 2558, 'whites': 2559, 'whiter': 2560, 'spreading': 2561, 'drab': 4074, 'motley': 2321, 'legged': 2563, 'prepares': 2565, 'olarge': 2566, 'stippling': 2567, 'rising': 2208, 'whiten': 2569, 'strip': 2570, 'tilted': 3175, 'sades': 2572, 'your': 2574, 'whitee': 2575, 'snakes': 1307, 'stare': 2577, 'grwon': 2578, 'ared': 2579, 'breaste': 4078, 'log': 2581, 'her': 2582, 'area': 2583, 'aren': 2584, 'gleam': 1236, 'strictly': 2586, 'there': 2587, 'topknot': 1244, 'rectries': 304, 'stark': 2590, 'sides': 4079, 'start': 2592, 'stealth': 2593, 'ares': 1345, 'lot': 2595, 'exceptional': 1629, 'viscous': 2597, 'tthis': 3142, 'egg': 4281, 'pollinator': 2472, 'complete': 2602, 'psotted': 2603, 'enough': 2604, 'proportionately': 2605, 'gliding': 2606, 'coloredd': 2607, 'buggy': 2608, 'eyrings': 2609, 'eyeliner': 2610, 'strikes': 2611, 'change': 2612, 'aquamarine': 2613, 'metalic': 193, 'exact': 4615, 'brain': 4849, 'wita': 2616, 'jut': 2617, 'drilling': 5113, 'pitched': 2620, 'blueish': 2621, 'with': 2622, 'handsome': 2623, 'wuth': 2624, 'j3t': 672, 'darken': 2626, 'arranged': 2627, 'largest': 5066, 'scecondaries': 2628, 'animal': 4400, 'blely': 2629, 'unveils': 2630, 'trices': 4992, 'strawberry': 2632, 'smallt': 2633, 'peeking': 2634, 'eyeballs': 1774, 'tones': 2636, 'dvd': 4087, 'site': 5041, 'wiith': 591, 'grass': 2639, 'torso': 2640, 'rust': 2641, 'darker': 2642, 'aa': 1626, 'upgoing': 2644, 'ohn': 2645, 'palate': 2646, 'accentuated': 836, 'ad': 2648, 'creat': 2649, 'angled': 4989, 'holes': 3835, 'accents': 2651, 'describe': 2652, 'am': 2653, 'avian': 2247, 'deep': 2655, 'fellow': 2656, 'blurred': 2131, 'as': 2659, 'vidid': 506, 'at': 2661, 'aw': 2662, 'walks': 2663, 'trace': 2664, 'supercillary': 2665, 'gradients': 2666, 'moves': 2667, 'spends': 2668, 'plumpish': 2669, 'ohs': 145, 'again': 2672, 'extending': 3912, 'inflated': 3192, 'comped': 2674, 'separating': 4534, 'farther': 1028, 'aforest': 3891, 'tight': 2677, 'narrowing': 2678, 'stripping': 2679, '5': 2680, 'someones': 2682, 'starkly': 2683, 'you': 2684, 'spreaded': 5418, 'strokes': 4552, 'separate': 2688, 'oenanthe': 2253, 'teeth': 2691, 'whiskers': 2692, 'm': 2693, 'whistling': 2694, 'includes': 2695, 'beakl': 2696, 'stocky': 2697, 'dominates': 3198, 'flattened': 2700, 'peak': 2701, 'suited': 2340, 'included': 2704, 'oragnge': 1758, 'srtiped': 2564, 'awkwardly': 2707, 'brandt': 2708, 'resembled': 1041, 'throaqt': 2711, 'berries': 4095, 'beast': 5083, 'bilateral': 2713, 'calls': 2714, 'blackx': 3203, 'thorn': 2715, 'mask': 2716, 'variegated': 2717, 'earthtones': 1934, 'curvy': 2719, 'mimic': 2720, 'hummingnird': 777, 'mass': 1203, 'resembles': 2723, 'evenly': 4098, 'starting': 2725, 'fingerlike': 5343, 'wingsthat': 5395, 'wedge': 4359, 'medoum': 2727, 'outlining': 11, 'looped': 2729, 'angles': 4998, 'pointing': 2730, 'breadth': 2731, 'brease': 4381, 'excepting': 2732, 'lack': 2734, 'acoss': 2735, 'riangular': 2736, 'bape': 2737, 'vill': 2738, 'sunk': 69, 'abdomon': 3511, 'bulb': 5000, 'messy': 2741, 'lacy': 2742, 'sideburns': 2743, 'larsuses': 2744, 'mulitcolored': 114, 'dawning': 4102, 'content': 463, 'eyerink': 2748, 'batches': 4386, 'fits': 4417, 'chain': 2750, 'hunting': 2751, 'sparkle': 2752, 'lining': 2753, 'nears': 4376, 'acting': 3206, 'sparkling': 189, 'good': 3135, 'tn': 2756, 'to': 2757, 'tail': 2758, 'present': 5002, 'undiscernable': 975, 'ti': 2761, 'crimson': 2762, 'smile': 2763, 'tailfeathers': 2764, 'te': 237, 'crest': 2766, 'cilliary': 2767, 'puffs': 2768, 'trim': 2769, 'liter': 273, 'recurved': 2771, 'brindled': 3210, 'song': 286, 'far': 291, 'ticked': 299, 'puffy': 2776, 'bubbled': 312, 'fat': 2778, 'coloured': 2779, 'print': 3212, 'plaid': 5007, 'stipes': 338, 'fal': 2782, 'fall': 2783, 'expansive': 3220, 'stacked': 2784, 'difference': 2786, 'supericiliary': 2687, 'exotic': 1858, 'fae': 383, 'oragne': 2790, 'bhird': 2791, 'cable': 2793, 'laying': 2264, 'joined': 2798, 'cascading': 2799, 'taled': 2084, 'large': 2801, 'blakc': 4924, 'anad': 460, 'dusky': 2804, 'variances': 2477, 'small': 2806, 'neckless': 2808, 'meduim': 2809, '20th': 2811, 'teh': 2812, 'entires': 2813, 'humongous': 2814, 'tel': 2816, 'ten': 1224, 'colossal': 2818, 'tea': 1600, 'eyelash': 3179, 'streets': 1044, 'tricolor': 2822, 'nearing': 2920, 'throughly': 2824, 'past': 2825, 'intricate': 3016, 'design': 2827, 'displays': 2828, 'pass': 3777, 'abdomenal': 2831, 'further': 2832, 'rectangular': 5016, 'chair': 2835, 'snippets': 2836, 'addition': 4970, 'will': 5017, 'bodwy': 643, 'swampy': 2839, 'abd': 2840, 'brien': 2841, 'spry': 1514, 'eyespot': 2843, 'deeply': 669, 'imperceptible': 1221, 'belyl': 2846, 'transforms': 2847, 'musty': 1327, 'burgundy': 2849, 'purpleish': 2850, 'thorny': 2851, 'version': 2852, 'flattish': 2855, 'neutrally': 2232, 'disbursed': 2857, 'seemed': 4114, 'racing': 2858, 'racoon': 2859, 'multitude': 751, 'movement': 2861, 'revealing': 2862, 'scowling': 2863, 'ochre': 3904, 'toes': 775, 'ferocious': 1810, 'tarsus': 5022, 'malaria': 2867, 'loose': 2869, 'yellower': 2133, 'ranges': 2871, 'wild': 5024, 'uneven': 2103, 'toed': 2874, 'arch': 2875, 'subcullaries': 5197, 'trunk': 2876, 'directions': 1100, 'grape': 2878, 'burd': 2879, 'yellowed': 2411, 'patterened': 2881, 'mult': 879, 'hass': 2883, 'fatter': 2884, 'search': 2885, 'striationsnon': 2886, 'bulged': 904, 'ahead': 2888, 'rumped': 3228, 'identifiable': 2890, 'redding': 2892, 'stretching': 2893, 'allows': 934, 'tial': 2895, 'creepy': 2896, 'miniature': 2898, 'amount': 960, 'base': 3229, 'milky': 980, 'superciliaries': 2902, 'formation': 2903, 'ayellow': 2904, 'narrow': 2905, 'ppatch': 2906, 'elongated': 2908, 'point': 5139, 'uniquely': 1014, 'nondistinct': 3788, 'followed': 2911, 'family': 2912, 'plain': 5011, 'secondarily': 3139, 'brush': 5298, 'hrey': 253, 'layer': 5031, 'tracking': 2916, 'broand': 2917, 'put': 3233, 'chunky': 2919, 'thought': 5103, 'snub': 2921, 'darkens': 2923, 'golf': 3430, 'smal': 2925, 'haunting': 3741, 'featers': 2926, 'qith': 2927, 'ash': 3236, 'eye': 2929, 'takes': 2930, 'slanty': 2931, 'taloned': 1172, 'beginning': 3237, 'faint': 1186, 'direction': 2936, 'two': 2937, 'shite': 2938, 'rustic': 2939, 'pearl': 2940, 'eys': 2941, 'splash': 2942, 'breaset': 2943, 'characteristics': 2292, 'taken': 1238, 'pupils': 4710, 'beloly': 1252, 'stretchy': 2948, 'firery': 2949, 'minor': 2950, 'wingtips': 1269, 'chestnut': 2952, 'flat': 2953, 'reacing': 2954, 'diamond': 2955, 'flap': 2956, 'pads': 2957, 'patsches': 2958, 'grabbing': 1308, 'crested': 2960, 'ckeek': 2961, 'marks': 2962, 'downlike': 1326, 'minuscule': 2964, 'blat': 2965, 'triangle': 3468, 'separative': 2966, 'blanding': 367, 'stick': 2968, 'varying': 1357, 'browned': 1361, 'avoce': 2971, 'mellow': 693, 'mounted': 2973, 'lemony': 2974, 'secondaies': 5015, 'strioed': 2977, 'town': 2978, 'keeping': 2980, 'pleasing': 2975, 'fore': 2982, 'hour': 2983, 'cluster': 2984, 'umber': 2985, 'scrawny': 2986, 'malt': 2987, 'wingbarred': 2988, 'blak': 2298, 'intercepted': 2990, 'dramatically': 4133, 'besides': 1831, 'citrine': 5323, 'mall': 2993, 'bron': 2139, 'striped': 4842, 'fronted': 2995, 'learn': 2124, 'abandon': 2997, 'dwith': 2998, 'seagull': 2999, 'male': 1511, 'specialized': 3001, 'notching': 3002, 'pick': 2901, 'beautiful': 3003, 'shark': 1522, 'palest': 3005, 'ewll': 3006, 'cown': 3007, 'taunt': 5162, 'cowl': 1544, 'ebelly': 3009, 'lightbrown': 3010, 'share': 3011, 'dusting': 4995, 'autumn': 3012, 'sphere': 3013, 'salty': 3014, 'scavenger': 2303, 'pond': 1617, 'bigbill': 1621, 'dhead': 3245, 'larhe': 1632, 'narrowly': 3022, 'huge': 3023, 'banding': 3024, 'awkward': 3025, '2white': 1670, 'brownb': 1674, 'conical': 5051, 'stripe': 3739, 'tubby': 1398, 'molds': 4588, 'glowing': 3032, 'heck': 3033, 'stemming': 3034, 'piebald': 1000, 'talons': 4139, 'blueand': 3037, 'fleck': 2785, 'strong': 2880, 'brownw': 3039, 'stip': 1770, 'creature': 3041, 'upto': 4275, 'blacked': 3043, 'plant': 3044, 'lanky': 1792, 'wirth': 3046, 'petite': 3047, 'thougout': 1813, 'soda': 3050, 'variant': 3051, 'winglay': 3052, 'pinyon': 3053, 'foled': 2346, 'blotted': 1840, 'catalog': 3056, 'fluffed': 3057, 'lighting': 3058, 'blood': 1857, 'offset': 5252, 'blacker': 1864, 'backyard': 4776, 'horizontal': 3062, 'flutter': 3063, 'pinted': 1897, 'breastss': 3065, 'a': 3066, 'belied': 4760, 'bluelegs': 1919, 'refined': 5033, 'orchard': 3069, 'resemble': 3070, 'coat': 3072, 'boasting': 4421, 'bugs': 217, 'dragon': 3075, 'roundish': 3076, 'generic': 1484, 'shore': 3078, 'shrap': 3079, 'wingset': 1721, 'vareigated': 1995, 'pay': 3082, 'eyeling': 3083, 'eyeline': 3084, 'breads': 2014, 'breadt': 300, 'weings': 3087, 'redish': 1406, 'infant': 2045, 'rounded': 3090, 'eyebrowns': 3091, 'earing': 1738, 'oh': 3093, 'stains': 1147, 'bowl': 3095, 'argyle': 3096, 'ite': 3097, 'soon': 3098, 'auburn': 3099, 'winglbars': 3100, 'manilla': 2106, 'rounder': 3102, 'wheeling': 3103, 'scott': 138, 'goldish': 3105, 'filters': 3106, 'soot': 3107, 'orance': 3108, 'concentric': 3109, 'cheated': 3110, 'its': 3111, 'roots': 2155, 'brid': 3113, 'highlighted': 5063, 'style': 3114, '20': 2177, 'graywith': 3116, 'bespeckled': 3117, 'areblack': 3118, 'alsos': 3119, 'lateral': 3120, 'greatly': 2543, 'bosom': 3121, 'fethers': 4723, 'actually': 3122, 'toupee': 3124, 'hillsides': 3125, 'parts': 5065, 'seagoing': 2681, 'squat': 5369, 'sunken': 3128, 'breastfeathers': 2262, 'cutie': 3130, 'might': 3131, 'wingspan': 3132, 'finer': 1509, 'brwon': 3134, 'ant': 1959, 'someone': 2299, 'sucking': 3136, 'nestled': 3137, 'was': 1933, 'food': 1690, 'synonymous': 3140, 'maler': 3141, 'transitioning': 4657, 'ye': 2338, 'predator': 3144, 'sprimarly': 3145, 'eyestripe': 3264, 'football': 3147, 'witting': 3061, 'frizzy': 3149, 'motled': 3150, 'iridescent': 3151, 'foot': 3152, 'lightens': 3153, 'bigger': 3154, 'mesmerizing': 4159, 'easily': 3156, 'rred': 3157, 'yellowy': 2449, 'wash': 3159, 'fully': 3160, 'settings': 3161, 'ashen': 563, 'goofy': 3164, 'slicked': 2504, 'predators': 3166, 'supercilliary': 3167, 'yeloow': 2511, 'shocking': 3169, 'harboring': 3170, 'characterized': 402, 'airsack': 3172, 'heavy': 2553, 'bulbous': 5217, 'shock': 4162, 'hark': 1026, 'restless': 3176, 'weight': 3177, 'interweaving': 3074, 'blaack': 1260, 'house': 3180, 'tiffany': 2598, 'hard': 3182, 'positioned': 3183, 'bronzish': 3184, 'poodle': 3185, 'blackwith': 3186, 'carion': 5375, 'extended': 2334, 'expect': 3188, 'carry': 1874, 'blues': 3191, 'beyond': 2673, 'immensely': 1251, 'really': 3194, 'waterbound': 3195, 'silky': 3196, 'admist': 3197, 'flower': 2699, 'podge': 3199, 'closest': 2337, 'blacks': 3200, 'surrounded': 3201, 'crowing': 1767, 'midsection': 3067, 'inferior': 4819, 'splocthes': 5076, 'yyellow': 2754, 'streaking': 3207, 'safety': 3208, 'hill': 3209, 'blackc': 2772, 'paradise': 3211, 'denoted': 2780, 'flame': 3213, 'deark': 2787, 'rectories': 3215, 'blacki': 3216, 'mutilcolored': 3217, 'discreetly': 4735, 'flesh': 5119, 'ass': 3219, 'bellyw': 5005, 'pun': 2823, 'protruding': 3221, 'dirt': 3222, 'squished': 3223, 'parched': 120, 'combed': 3225, 'extraordinary': 2889, 'abnormally': 2900, 'brightest': 3231, 'disposition': 3232, 'backed': 2918, 'heads': 3281, 'nexck': 2924, 'rises': 2928, 'obre': 2933, 'wngs': 3238, 'bushes': 330, 'definition': 3240, 'albatross': 3241, 'choppy': 4170, 'crimple': 4176, 'fashioning': 5062, 'dip': 3243, 'running': 2343, 'round': 1433, 'deather': 635, 'browinsh': 3247, 'diffuses': 3248, 'w': 3249, 'leading': 3284, 'orage': 3251, 'beaver': 3017, 'triangualr': 3252, 'solidly': 3254, 'feed': 3255, 'copper': 3256, 'smoother': 3286, 'emblazoned': 3260, 'feel': 3261, 'indeterminate': 1697, 'number': 3146, 'fancy': 3265, 'feeder': 3266, 'feet': 3267, 'mistly': 3189, 'enlarged': 3269, 'river': 3270, 'dominating': 3168, 'hangs': 3272, 'general': 5001, 'flycatcher': 57, 'bland': 3274, 'yellownape': 4175, 'blacktail': 3276, 'tana': 4276, 'splattered': 3277, 'causing': 3798, 'koi': 3278, 'horse': 3279, 'fuller': 3234, 'jet': 3282, 'become': 3283, 'interwoven': 3250, 'wingtripes': 3253, 'swipe': 3258, 'moderately': 3287, 'neatly': 1998, 'elswhere': 3289, 'intense': 3905, 'tippe3d': 3290, 'pronouncedly': 3291, 'wonderful': 3292, 'seafaring': 3293, 'storm': 3294, 'trails': 3295, 'periwinkle': 3296, 'hummingbird': 2351, 'banana': 3299, 'bumble': 3302, 'bowed': 3303, 'absent': 3304, 'heada': 3305, 'relationship': 3306, 'behind': 3307, 'eyewing': 3308, 'contains': 2935, 'needle': 3309, 'immediate': 3310, 'sizable': 3311, 'cylindrical': 2873, 'pars': 3313, 'cheekbones': 3314, 'priimaries': 3315, 'sturdily': 2689, 'printed': 3318, 'clored': 3319, 'graywings': 3320, 'rectises': 3321, 'abdoment': 1972, 'peaked': 3323, 'king': 3324, 'kind': 3325, 'scheme': 3298, 'compliments': 3327, 'bluejay': 3328, 'double': 3329, 'intermittent': 3330, 'typically': 3331, 'grey': 3332, 'dogs': 2237, 'tummy': 3334, 'determined': 3335, 'cuff': 557, 'myriad': 4816, 'cross': 5061, 'elaborate': 3338, 'seamlessly': 583, 'toward': 3340, 'playing': 3341, 'appearance': 5014, 'luster': 5405, 'dried': 3343, 'expnasive': 3344, 'teensy': 3345, 'eyestrip': 3346, 'pigeon': 3202, 'person': 3756, 'basically': 3348, 'well': 5101, 'mossy': 3350, 'alongside': 3351, 'familiarizes': 3352, 'randomly': 3353, 'identifier': 5260, 'substantial': 3354, 'whiite': 3355, 'blond': 3356, 'dullish': 3357, 'carrys': 2963, 'concentration': 3359, 'oriole': 3360, 'fading': 3361, 'winfs': 3362, 'lie': 3363, 'bake': 2286, 'tarsuses': 3365, 'winspan': 1854, 'flared': 3367, 'patterns': 5104, 'btight': 5434, 'self': 3370, 'flaming': 3371, 'camouflage': 3372, 'silvery': 3373, 'pluamge': 3374, 'flares': 1580, 'entirety': 3376, 'tarsused': 2306, 'build': 3378, 'breaststroke': 3379, 'ancillary': 3380, 'longish': 724, 'buff': 3383, 'unremarkable': 3384, 'serene': 3385, 'towards': 3386, 'electric': 3388, 'eyelings': 3389, 'coppers': 5312, 'quote': 1594, 'eater': 3392, 'mottle': 5108, 'reach': 268, 'ized': 3395, 'raucous': 540, 'complementing': 3397, 'significant': 3398, 'position': 3399, 'nothing': 3400, 'mangrove': 3401, 'karsus': 3403, 'extremely': 3404, 'varrying': 3405, 'stately': 2181, 'patterned': 3408, 'ridged': 3409, 'chested': 3410, 'dissolves': 3412, 'bbird': 2619, 'clear': 3415, 'cover': 3416, 'glossier': 4927, 'barred': 3418, 'part': 3312, 'clean': 755, 'patternes': 3421, 'organish': 3422, 'barrel': 3423, 'notch': 3424, 'whiteand': 2685, 'tellow': 3426, 'lying': 3427, 'stones': 3428, 'whitebody': 1744, 'particularly': 1204, 'gold': 3431, 'sparrow': 3432, 'blended': 3434, 'fins': 3435, 'scattered': 3436, 'taller': 5116, 'perches': 3438, 'hillside': 3439, 'focusing': 3316, 'primaties': 5429, 'relation': 3440, 'sliver': 3725, 'impressively': 3441, 'stalker': 3442, 'perched': 3443, 'stubbybeak': 3445, 'wingbars': 4206, 'fine': 3446, 'find': 3447, 'enormously': 5117, 'whited': 4557, 'wraps': 2145, 'believe': 3449, 'unusable': 3450, 'dividing': 3451, 'crawl': 2187, 'camel': 1476, 'distributed': 3454, 'lard': 3883, 'thoughout': 3456, 'cavorts': 2043, 'slant': 5246, 'explodes': 3458, 'pretty': 3459, 'money': 4932, 'diwth': 3461, 'circle': 3462, 'flightless': 3463, 'stripings': 700, 'gret': 3466, 'his': 3467, 'brownback': 2143, 'upstanding': 3469, 'gratis': 3470, 'dandelion': 1537, 'wints': 3472, 'wingsbar': 3473, 'breask': 3474, 'smell': 2452, 'sunny': 3476, 'frumpy': 3477, 'trees': 3478, 'speckled': 3479, 'blank': 3480, 'longorange': 3481, 'grew': 3326, 'hid': 3483, 'breast': 3484, 'closely': 3485, 'combinations': 1667, 'during': 3487, 'camoflagued': 3488, 'silk': 3489, 'him': 3490, 'maroon': 5381, 'elf': 3492, 'chick': 5193, 'laterally': 3493, 'fluff': 3494, 'catcher': 537, 'belled': 4504, 'posseses': 3497, 'withe': 3498, 'spikey': 3499, 'matted': 3500, 'witha': 3501, 'eithe': 3502, 'bird': 4217, 'blakened': 2378, 'spiked': 3504, 'common': 2789, 'sclera': 3506, 'smll': 3508, 'bella': 3031, 'withw': 3510, 'belley': 530, 'blazes': 5003, 'blac': 3512, 'withr': 3513, 'larsi': 1917, 'scanning': 2168, 'bars': 3516, 'set': 3517, 'stump': 3518, 'proximal': 182, 'curbved': 3520, 'llarge': 3521, 'jade': 143, 'pnted': 3523, 'sti': 3524, 'paler': 3525, 'slanting': 3526, 'see': 3527, 'individual': 3528, 'dumb': 1515, 'ornage': 589, 'sea': 3531, 'clutching': 4805, 'squarish': 353, 'origami': 3491, 'bark': 3535, 'hole': 3536, 'arm': 1698, 'mulitiple': 3538, 'barn': 3539, 'glistening': 3541, 'degree': 4223, 'halved': 3543, 'gree': 3544, 'muted': 3545, 'satin': 2391, 'wow': 3547, 'currently': 3548, 'bellied': 3549, 'rown': 3968, 'rejecting': 3550, 'drooping': 3551, 'fans': 2197, 'creme': 3554, 'various': 3555, 'forked': 3556, 'bulging': 4779, 'downturn': 1575, 'halfway': 2393, 'nope': 3559, 'compliment': 3560, 'ings': 3561, 'ehite': 2394, 'makeup': 3562, 'recently': 3563, 'hthe': 3564, 'missing': 3566, 'sold': 3567, 'attention': 3568, 'bodt': 4229, 'ligth': 3569, 'spray': 3570, 'blots': 3204, 'withgrey': 3572, 'invisible': 3573, 'striated': 3574, 'spalshes': 3575, 'aregray': 514, 'gren': 3339, 'bronze': 3578, 'abruptly': 3579, 'brillant': 3580, 'both': 3581, 'last': 3582, 'reverse': 67, 'stipe': 4066, 'tapered': 3584, 'barely': 3585, 'bandit': 3587, 'riddled': 3588, 'blazed': 3589, 'yelow': 4311, 'alot': 3590, 'breasy': 4448, 'whole': 3593, 'finds': 3594, 'baltimore': 5106, 'thickbill': 3596, 'flies': 3597, 'patchs': 3599, 'iwth': 3945, 'mini': 3301, 'simple': 3601, 'sweet': 3602, 'poiny': 3603, 'others': 4236, 'vent': 4948, 'sweep': 3604, 'lake': 5141, 'steely': 3606, 'bely': 3608, 'simply': 3259, 'littl': 2404, 'billy': 3611, 'graying': 3612, 'bench': 2807, 'frosting': 1900, 'belt': 3615, 'startling': 3616, 'dul': 3617, 'napr': 3618, 'dun': 3619, 'duo': 3620, 'help': 3621, 'damp': 3622, 'due': 3623, 'naylor': 2236, 'runner': 711, 'nape': 3626, 'describes': 284, 'pf': 3390, 'evenness': 2007, 'pearlescent': 3630, 'brick': 3631, 'colorings': 3632, 'ahwite': 5354, 'chiseled': 3634, 'nesting': 3088, '4': 3635, 'firm': 3123, 'scalp': 2615, 'flight': 3639, 'squirrel': 3640, 'gay': 3641, 'fire': 3642, 'wingstripes': 3643, 'hind': 3644, 'gas': 3645, 'great': 4244, 'else': 3647, 'moment': 3736, 'adaptable': 32, 'rellow': 3148, 'bight': 3649, 'lives': 3650, 'systematic': 3651, 'wrings': 4987, 'andcovets': 3653, 'towns': 3654, 'interior': 3655, 'poised': 3656, 'plants': 48, 'intriguing': 3658, 'fur': 62, 'noticeable': 1498, 'seconaries': 73, 'caged': 3663, 'straight': 3664, 'bill': 81, 'grainy': 3666, 'batch': 3667, 'downwoards': 3668, 'snubbed': 1499, 'while': 3671, 'sideand': 3672, 'skies': 3673, 'sixed': 3674, 'distinquished': 3675, 'ful': 3676, 'sma': 3677, 'ebrown': 161, 'ciliary': 3679, 'pylon': 2479, 'grip': 3138, 'erupting': 3682, 'sizeable': 3683, 'squinty': 3684, 'arching': 3685, 'moth': 3686, 'shades': 4248, 'dotting': 3688, 'rectrrices': 5245, 'alrge': 3690, 'rin': 3691, 'cornw': 3692, 'itself': 1002, 'esque': 4249, 'beautifully': 3694, 'ready': 3695, 'hellow': 3696, 'cheekpatch': 3697, 'greay': 4250, 'neutral': 5341, 'kernal': 289, 'frown': 3700, 'shar': 302, 'funny': 3702, 'lengthy': 3703, 'since': 3661, 'shorted': 3705, 'coloringful': 3706, 'leafy': 615, 'awhite': 3915, 'undersides': 3709, 'rosy': 3710, 'elevated': 356, 'shorter': 3713, 'superciliary': 4817, 'emphasize': 2248, 'pecking': 392, 'grand': 3717, 'greyed': 3718, 'sprouted': 3719, 'eybrow': 3720, 'undersided': 411, 'gace': 2194, 'composition': 1064, 'eybros': 3724, 'higher': 425, 'andinner': 3726, 'used': 441, 'roound': 4694, 'sppotted': 3728, 'stripy': 3729, 'cigar': 3730, 'leopard': 3731, 'covart': 3732, 'wingtipped': 2448, 'alert': 3734, 'raccoon': 1508, 'flurry': 2081, 'flown': 1766, 'uses': 513, 'purpose': 516, 'peachy': 3740, 'geeenish': 1472, 'rectrcies': 533, 'assortment': 3743, 'brome': 3745, 'noticeably': 2854, 'darkening': 4383, 'retracted': 3366, 'frozen': 3748, 'lower': 3749, 'mollusks': 3750, 'patchees': 3751, 'cheek': 3752, 'chickens': 604, 'center': 5094, 'wedged': 3754, 'segmented': 622, 'boat': 3757, 'chees': 631, 'unspotted': 3759, 'covets': 646, 'longand': 653, 'flowing': 4212, 'brasts': 3762, 'clasped': 3532, 'wig': 3765, 'watching': 3766, 'facial': 3768, 'lbird': 3769, 'checkering': 3770, 'styled': 3771, 'fraille': 3772, 'triagles': 3773, 'chesk': 3342, 'distracting': 900, 'shape': 745, 'openly': 2897, 'rectricites': 527, 'elegance': 493, 'using': 3780, 'useful': 3781, 'stalk': 3336, 'metallic': 3783, 'forrest': 3784, 'cocks': 3785, 'posteriorly': 4266, 'pairs': 4316, 'cut': 3787, 'roused': 2002, 'hued': 3789, 'majority': 3375, 'smaler': 3791, 'spills': 2721, 'breastline': 841, 'squiggle': 3707, 'exclusively': 3794, 'vague': 3795, 'dhas': 3796, 'eager': 2382, 'circling': 3779, 'lifts': 2573, 'location': 887, 'tuxedo': 3800, 'throate': 3801, 'extend': 4891, 'mod': 2882, 'easter': 3804, 'excited': 63, 'surprised': 3806, 'caramel': 3807, 'jutting': 750, 'ripples': 4631, 'nwings': 3809, 'lacking': 3810, 'vertically': 3811, 'reptilian': 965, 'bid': 970, 'grann': 3723, 'bib': 3815, 'whispy': 3816, 'ruff': 991, 'pattering': 3818, 'falcon': 1030, 'bit': 3821, 'bloody': 3822, 'bir': 3823, 'bis': 1043, 'translucent': 3825, 'formal': 3826, 'glittery': 3827, 'scaled': 4272, 'd': 3829, 'semi': 1085, 'mans': 3496, 'puffing': 3381, 'wingbards': 1097, 'criss': 3834, 'bellyt': 2777, 'shapes': 3836, 'collect': 3591, 'hjas': 3838, 'encasing': 2525, 'borrows': 3840, 'vibrant': 5181, 'striples': 3842, 'chesy': 3843, 'abdobmen': 1729, 'healthy': 1999, 'larsus': 3846, 'borad': 5340, 'trailing': 5391, 'wbrown': 3848, 'caspian': 1200, 'often': 3850, 'hinted': 3851, 'heafdis': 3852, 'mauve': 3853, 'fathers': 3854, 'wingulars': 3855, 'some': 1232, 'insulated': 1663, 'thighss': 3858, 'underdeveloped': 3859, 'streaming': 2157, 'strongest': 3861, 'pack': 3680, 'palm': 3862, 'supercilials': 2457, 'sight': 3864, 'underbelly': 3865, 'curious': 1287, 'oddly': 3867, 'pale': 3868, 'hblack': 4614, 'pronounced': 3869, 'proprtionally': 1324, 'cpffee': 2162, 'amongst': 4742, 'yellowbody': 3872, 'dull': 3986, 'pet': 3873, 'boasts': 4280, 'shall': 3876, 'eand': 3877, 'extends': 5147, 'per': 3878, 'contrast': 2860, 'medum': 1094, 'connecting': 3880, 'lare': 74, 'abodomen': 3391, 'secondiaries': 3884, 'excluding': 5178, 'acrown': 3722, 'temple': 1423, 'scaly': 1431, 'be': 1438, 'eyeriing': 1448, 'restices': 4193, 'breastplate': 3889, 'bellys': 5008, 'bl': 31, 'salmon': 3394, 'continuing': 3892, 'lakes': 1482, 'cronw': 3894, 'stem': 3895, 'bliue': 3896, 'nelly': 3897, 'step': 3819, 'assists': 3899, 'concealed': 3900, 'upraised': 1521, 'wingsbars': 3903, 'crone': 763, 'by': 453, 'aswell': 3906, 'sunbathing': 3907, 'dormant': 1561, 'shadings': 3909, 'mingled': 1578, 'anything': 3911, 'most': 3396, 'exaggerated': 1591, 'secodaries': 2899, 'slamm': 3914, 'birdbath': 1615, 'page': 5196, 'torpedo': 4905, 'range': 3916, 'coif': 1661, 'breasted': 3918, 'winga': 3919, 'complimentary': 867, 'fluttering': 1664, 'angry': 1543, 'block': 3923, 'anterior': 3924, 'oraneg': 5198, 'wiphite': 3925, 'real': 3926, 'moss': 3927, 'shstriped': 1718, 'anda': 3930, 'into': 3931, 'within': 3932, 'inkly': 3933, 'encircling': 3934, 'framed': 3935, 'sixths': 3936, 'appropriate': 1765, 'clack': 3938, 'primarily': 3939, 'lender': 3940, 'sueprciliary': 1785, 'proportional': 3942, 'birched': 3943, 'variously': 2257, 'chirping': 1803, 'plummage': 3086, 'nexk': 4296, 'bicolored': 3948, 'rows': 3949, 'span': 3950, 'propel': 5202, 'shading': 3820, 'few': 4297, 'transcending': 3955, 'exposed': 3956, 'lone': 3957, '81043892': 3958, 'fast': 1871, 'abodmen': 3960, 'indoor': 4782, 'roadrunner': 3961, 'lithe': 3962, 'suit': 3963, 'forward': 3964, 'tailgaters': 5333, 'bored': 3966, 'sections': 3967, 'opens': 1687, 'cheeky': 3969, 'fragmented': 1951, 'covertail': 2810, 'lmostly': 3387, 'an': 3973, 'elsewhere': 3974, 'bore': 1117, 'recognized': 5267, 'dips': 1989, 'retina': 3977, 'stunning': 5019, 'disproportional': 3978, 'hread': 3979, 'toucan': 3018, 'indistinguishable': 3981, 'upright': 2049, 'innter': 3983, 'line': 3984, 'topline': 2063, 'calico': 2069, 'raising': 3987, 'penguin': 3988, 'fades': 3989, 'whtie': 5208, 'aq': 3991, 'creams': 3992, 'consist': 2105, 'ligjt': 5184, 'u': 5412, 'apricot': 2117, 'sturdy': 3414, 'up': 3998, 'horned': 321, 'oily': 4304, 'roudned': 4000, 'reflects': 4305, 'irridescent': 2169, 'highlight': 4004, 'intently': 4006, 'throut': 481, 'covet': 4005, 'sticklike': 4009, 'bell': 4010, 'underbellied': 2205, 'v': 2996, 'discolored': 4013, 'adults': 2224, 'necka': 2326, 'defined': 4016, 'metal': 1598, 'sepia': 4019, 'doesn': 4021, 'tinging': 4022, 'lavendar': 2273, 'single': 4025, 'skimming': 2279, 'additionally': 5428, 'chevron': 3920, 'undertones': 4028, 'obtuse': 4029, 'graceful': 4030, 'curl': 4031, 'work': 1435, 'underwings': 4032, 'oval': 4278, 'cotton': 2327, 'reast': 4034, 'rainbow': 4035, 'tari': 4036, 'lemon': 4037, 'peach': 2354, 'tuft': 4039, 'amounts': 4040, 'dashed': 1285, 'malicious': 4042, 'fears': 4043, 'backs': 4044, 'rectines': 5440, 'gry': 4046, 'occasional': 4047, 'nick': 4048, 'swath': 2420, 'definitely': 4050, 'primariaries': 4051, 'tuff': 4052, 'dimmer': 2433, 'whings': 4054, 'ar': 2660, 'vireo': 2219, 'nice': 4056, 'samll': 4057, 'slick': 2494, 'magestic': 4058, 'conventionally': 3402, 'gloss': 4060, 'wrinkly': 4061, 'tarsi': 3775, 'subby': 4318, 'sided': 4063, 'faintly': 4064, 'breasts': 4065, 'redcrown': 2521, 'luxurious': 4067, 'helping': 4068, 'chirp': 4069, 'insect': 4070, 'webbing': 2487, 'allowing': 4072, 'intersparsed': 4073, 'breathed': 2070, 'crowwn': 355, 'withpink': 2488, 'chameleon': 4077, 'clue': 2336, 'desert': 2591, 'rprimaries': 2512, 'grayed': 5280, 'rectrice': 3790, 'e': 4082, 'delineation': 4083, 'wafer': 4084, 'cling': 3035, 'overtop': 4086, 'blend': 3425, 'crowbn': 3174, 'wards': 2650, 'extruding': 2207, 'andthe': 4091, 'fresh': 2703, 'underrated': 4094, 'topsides': 2712, 'hello': 744, 'creast': 4097, 'once': 2724, 'paddling': 4517, 'partial': 4101, 'gr': 2746, 'iridescence': 5425, '0ff': 4103, 'underbrush': 1812, 'nap': 4049, 'overly': 733, 'cheerful': 4107, 'legnth': 4108, 'tarusus': 4081, 'softer': 2803, 'classic': 2233, 'dainty': 4111, 'go': 2826, 'grays': 2830, 'bifurcated': 5227, 'subtly': 2328, 'bluish': 1868, 'aligning': 4116, 'abundant': 4117, 'lille': 3317, 'peering': 4119, 'rufous': 4120, 'young': 4121, 'corwn': 4122, 'nose': 4123, 'rusted': 5404, 'varigated': 4125, 'teardrop': 4126, 'withwhtie': 4127, 'helps': 2907, 'striping': 4129, 'alight': 450, 'details': 5409, 'gravel': 2698, 'statured': 4132, 'friendly': 2991, 'bottomed': 2000, 'browns': 3038, 'rockish': 4137, 'outside': 4138, 'dessert': 3036, 'continues': 4140, 'alder': 4141, 'startlingly': 4142, 'melding': 4143, 'spoon': 4144, 'wavy': 4145, 'spotty': 5023, 'mixing': 2599, 'gins': 4147, 'decorating': 4148, 'feaher': 3406, 'splattering': 4150, 'wipe': 4151, 'framing': 4152, 'posture': 4153, 'molar': 261, 'entire': 4155, 'crowed': 4158, 'tonal': 3155, 'stiff': 4160, 'wingbaar': 4620, 'array': 1876, 'having': 4096, 'ornately': 4163, 'flouted': 4164, 'tooth': 4165, 'mottled': 4166, 'swallow': 4945, 'second': 4968, 'crayon': 4168, 'lactus': 4169, 'exhibits': 4092, 'noted': 4171, 'hiding': 5257, 'strongly': 4173, 'smaller': 4174, 'gripping': 3275, 'attaches': 5224, 'their': 4632, 'alogn': 4177, 'crow': 4178, 'bronish': 4179, 'wrestling': 2388, 'rumps': 4023, 'rivers': 4182, 'velvety': 4183, 'hasa': 2870, 'spaecks': 4184, 'orrange': 2856, 'traveling': 4186, 'picked': 4187, 'earth': 4188, 'unattractive': 4189, 'bloodshot': 1465, 'ctow': 4191, 'odd': 4192, 'wings': 3824, 'blackcrown': 1281, 'yellowblack': 4196, 'plays': 4197, 'eyllow': 400, 'opaque': 4199, 'giving': 4200, 'bresat': 4201, 'birs': 4202, 'eyepatch': 4203, 'birt': 4204, 'paddle': 4205, 'alson': 3127, 'silverish': 4207, 'consistently': 4208, 'birk': 3455, 'colorization': 4210, 'eating': 3888, 'birl': 4211, 'greyis': 4563, 't': 2460, 'scissored': 4213, 'crossed': 4349, 'toenails': 4215, 'mediums': 3000, 'herringbone': 3954, 'yllow': 322, 'armpit': 4219, 'scenery': 3515, 'pinhole': 4222, 'stance': 3542, 'leg': 4224, 'beasts': 4226, 'stature': 4227, 'giant': 3448, 'nonexistent': 4230, 'iconic': 2517, 'secondairies': 3746, 'thic': 4352, 'foraging': 4233, 'swirled': 4234, 'separation': 150, 'making': 2466, 'aerodynamic': 4237, 'borwn': 4239, 'enitrely': 95, 'extreme': 2726, 'vividness': 142, 'bottle': 4243, 'convert': 3646, 'fuffy': 3008, 'crows': 4245, 'averages': 4354, 'larger': 4247, 'stalky': 3687, 'colorufl': 2419, 'redbreast': 3698, 'abdoman': 4251, 'spines': 4252, 'repel': 4253, 'thaty': 4254, 'fluffier': 3452, 'curvedb': 3735, 'crowns': 3652, 'shaded': 4258, 'makes': 4259, 'furrowed': 4260, 'surves': 4261, 'thats': 4262, 'composed': 782, 'barsus': 4264, 'bare': 3529, 'shovel': 4265, 'apple': 3786, 'nearest': 2631, 'shake': 4871, 'win': 4268, 'thin': 4350, 'anorange': 4270, 'irange': 5170, 'perky': 5276, 'ticking': 3882, 'sienna': 2485, 'hardy': 696, 'ran': 3625, 'wit': 4274, 'droops': 3847, 'duck': 4277, 'limb': 3860, 'yellwo': 4279, 'singing': 3874, 'headed': 3600, 'cloud': 4282, 'lime': 4283, 'unpleasantly': 1958, 'fed': 1648, 'fee': 4286, 'from': 4287, 'withbrown': 4288, 'usa': 3910, 'remains': 4290, 'kin': 4291, 'snacks': 4364, 'otter': 2079, 'cray': 4294, 'next': 4295, 'brested': 3947, 'zig': 3953, 'depicted': 4298, 'camera': 4299, 'exceptionally': 2894, 'iner': 4301, 'cram': 4302, 'irises': 4303, 'pleated': 3802, 'midday': 4001, 'touched': 708, 'sort': 4307, 'pencil': 1089, 'popped': 4309, 'naped': 4310, 'tarsusa': 3592, 'wades': 3744, 'tailbars': 4314, 'becomes': 4315, 'ith': 3092, 'elongaed': 1966, 'tangerine': 4062, 'supiciliary': 4596, 'bright': 2050, 'earthen': 4319, 'thatch': 4320, 'trail': 4321, 'carrying': 3522, 'napes': 4323, 'sharply': 4324, 'rectricies': 4325, 'baby': 4326, 'actual': 5261, 'billed': 4328, 'pieces': 4371, 'blackthroat': 4330, 'appointed': 4331, 'hints': 4332, 'fan': 2781, 'retries': 4334, 'wtih': 4335, 'crosses': 4336, 'ponted': 4337, 'arched': 4338, 'redand': 4339, 'f': 4340, 'this': 4341, 'bally': 4342, 'sbelly': 921, 'crocked': 4790, 'slit': 4374, 'dims': 4391, 'dusted': 4347, 'anywhere': 4348, 'obvious': 4214, 'rectangle': 2796, 'smear': 4351, 'highly': 4232, 'meet': 4353, 'grin': 3799, 'hash': 4355, 'thie': 4356, 'bend': 4357, 'tat': 4358, 'hip': 15, 'od': 4360, 'intermixing': 4361, 'bent': 4362, 'grace': 3457, 'do': 3565, 'process': 4365, 'fishtail': 4366, 'proportions': 3595, 'slim': 4368, 'tax': 4369, 'biggish': 4370, 'secondaires': 4329, 'wih': 4271, 'bones': 4380, 'dakr': 4346, 'crowbars': 4375, 'narsus': 1717, 'smattering': 3828, 'tab': 4378, 'tam': 4379, 'tal': 4373, 'hit': 2671, 'tan': 4382, 'pionted': 1754, 'covring': 4384, 'sis': 886, 'curving': 4388, 'varied': 701, 'sit': 4390, 'waterthrush': 3218, 'esp': 4088, 'waterproof': 3471, 'wingets': 3460, 'tai': 4394, 'regions': 4395, 'located': 4396, 'varies': 4397, 'forest': 4399, 'itsh': 3257, 'contrasts': 2005, 'aft': 1757, 'yellowpatch': 4402, 'blackhead': 4403, 'pointedbrown': 4404, 'medium': 1961, 'camoflauge': 4406, 'stock': 839, 'profile': 4408, 'yellowish': 4409, 'camoflague': 3782, 'bidds': 3475, 'designed': 4601, 'plump': 2082, 'act': 4411, 'cheeek': 3235, 'blocky': 4413, '3': 2152, 'walk': 4957, 'overaqall': 4414, 'demonstrating': 2018, 'bown': 4416, 'maske': 2580, 'da': 1481, 'blackfeet': 4419, 'glack': 4420, 'daring': 4389, 'pocket': 4422, 'chubbier': 726, 'purely': 4787, 'daggerlike': 4424, 'boday': 4425, 'light': 4426, 'spattered': 4700, 'wingband': 2122, 'cormorant': 1218, 'its0': 4429, 'accenting': 4430, 'scoop': 4431, 'whiteexcept': 4432, 'allow': 4433, 'skull': 3993, 'necklace': 4434, 'rouded': 3133, 'subsequently': 4435, 'lined': 4436, 'mustache': 4437, 'molted': 4438, 'least': 3288, 'saturated': 4441, 'thigh': 4442, 'haired': 5264, 'heater': 5229, 'bellty': 4444, 'minnow': 4445, 'move': 4446, 'equally': 5285, 'whilst': 1923, 'taluses': 5361, 'ecru': 3411, 'including': 4450, 'tourquoise': 4393, 'horns': 4452, 'porpotion': 4453, 'iswhite': 4454, 'hock': 4455, 'disgruntled': 4908, 'curling': 4456, 'eyereing': 444, 'superior': 3369, 'longitudinal': 2047, 'he': 5349, 'lo': 4461, 'relativity': 4462, 'bue': 4463, 'burgandy': 4464, 'shortened': 3681, 'prpportion': 4466, 'pipping': 4467, 'resat': 2797, 'blaks': 4470, 'pussy': 3767, 'orange': 4472, 'clusters': 3540, 'defining': 1449, 'image': 4475, 'speckle': 4198, 'designs': 4477, 'crucial': 3951, 'orangey': 4479, 'fourth': 2788, 'dab': 4481, 'orangebreast': 4482, 'organge': 4483, 'frontal': 4484, 'anchored': 2285, 'bat': 4486, 'oranges': 4487, 'blessed': 4488, 'withblack': 4489, 'dock': 4490, 'spikes': 3495, 'fiesty': 4492, 'snake': 4493, 'dar': 3586, 'hands': 2733, 'front': 4496, 'jaguar': 4497, 'caking': 4498, 'transparent': 3382, 'serpentine': 4500, 'peaceful': 4501, 'goldenrod': 4502, 'wnite': 4503, 'progressive': 4885, 'profound': 4505, 'tailtips': 4506, 'abulky': 4405, 'haed': 4508, 'narrown': 4509, 'identified': 4510, 'blazing': 4511, 'crudely': 4512, 'nicely': 731, 'delightful': 4514, 'bil': 3808, 'blacktop': 4516, 'eyeringed': 1055, 'crossing': 4518, 'feathersand': 2568, 'beneath': 4520, 'circular': 1489, 'gazing': 2979, 'treads': 4515, 'tot': 4524, 'lilac': 4525, 'nubby': 4526, 'bump': 4527, 'structure': 4080, 'chunk': 4529, 'grreen': 5300, 'undebody': 4530, 'inverted': 4531, 'whitish': 4532, 'supercilliaries': 4533, 'jagged': 3875, 'our': 4535, '80': 4536, 'transitions': 4537, 'tubular': 4538, 'sandy': 4539, 'out': 4540, 'poiinted': 3944, 'necked': 3503, 'offsetting': 4543, 'generalizations': 4544, 'bag': 4545, 'armor': 4546, 'res': 4547, 'smow': 4548, 'bad': 3347, 'wholly': 4550, 'stub': 4551, 'include': 4131, 'mate': 4553, 'latterally': 2369, 'rec': 61, 'prinaries': 4556, 'purplish': 2576, 'supports': 840, 'reg': 4559, 'red': 4560, 'oblong': 78, 'buds': 4562, 'mosty': 328, 'thrush': 4564, 'maize': 5306, 'spot': 1649, 'maintains': 127, 'diagonal': 4567, 'reectrices': 148, 'strap': 4569, 'ban': 4570, 'approaches': 4003, 'bire': 4216, 'collection': 4573, 'witht': 3509, 'greywith': 4575, 'retrieves': 4577, 'retriever': 4578, 'sweeps': 2989, 'canters': 222, 'poisoned': 2740, 'cured': 4581, 'g': 4582, 'could': 4584, 'metalli': 260, 'cheast': 4586, 'outerrectrices': 4587, 'times': 4589, 'happily': 4590, 'keen': 4591, 'superciiary': 4149, 'prmary': 4766, 'sround': 4593, 'broze': 4594, 'wading': 2120, 'camouflaging': 2585, 'shortish': 4597, 'today': 4598, 'receding': 4343, 'predominate': 678, 'upperparts': 4602, 'insane': 3514, 'streaksand': 4415, 'yelly': 365, 'widely': 3716, 'yellw': 4607, 'lon': 2588, 'possessing': 4609, 'cofee': 4460, 'powerful': 4611, 'delicately': 4612, 'reaches': 4613, 'sloped': 2589, 'yello': 409, 'coloful': 414, 'bends': 4618, 'flows': 4619, 'pincer': 436, 'eminence': 4621, 'peters': 4622, 'quality': 4623, 'strped': 4624, 'buoyed': 4625, 'scruffy': 3519, 'rushes': 5318, 'bull': 4999, 'hawk': 4418, 'laced': 4626, 'fascinating': 4627, 'ovenbird': 4610, 'thhis': 4628, 'boring': 4666, 'pots': 523, 'frontside': 4630, 'relations': 3300, 'attach': 3699, 'attack': 546, 'gradienting': 5320, 'balck': 4634, 'irdescent': 571, 'secondar': 4636, 'perfectly': 4637, 'six': 4440, 'hea': 2228, 'flume': 4640, 'intersperced': 4641, 'shell': 4642, 'accompany': 2524, 'shiny': 3921, 'big': 3812, 'shorts': 4645, 'completed': 4646, 'exactly': 641, 'drooped': 3793, 'shorty': 4648, 'feint': 4649, 'slides': 4650, 'slider': 4651, 'opalescent': 4652, 'regards': 4653, 'maze': 5325, 'rst': 4654, 'yellowing': 896, 'lush': 4656, 'slopes': 2596, 'filled': 4658, 'mangy': 4292, 'steel': 4660, 'greybird': 4018, 'photograph': 4662, 'surrounds': 5064, 'yuellow': 2114, 'clunky': 4664, 'colorded': 4665, 'split': 387, 'distinguishes': 4667, 'crownand': 3669, 'richer': 772, 'extraordinarily': 776, 'humped': 780, 'cartoonish': 4672, 'bdy': 4468, 'appealing': 4673, 'arc': 4674, 'lover': 810, 'identical': 5158, 'elongate': 815, 'stubby': 1949, 'indistinct': 4679, 'dusty': 3670, 'bodice': 4682, 'bard': 2690, 'exhibit': 4684, 'sripes': 4685, 'greenishyellow': 4686, 'lightly': 4687, 'throad': 875, 'splatter': 3227, 'brwn': 4690, 'grayish': 4691, 'blil': 897, 'curves': 4693, 'interrupted': 4428, 'gentle': 911, 'recessed': 4696, 'curven': 4697, 'sriped': 929, 'have': 4699, 'close': 3534, 'bluie': 940, 'curved': 945, 'darkpointed': 4703, 'flecking': 4704, 'border': 4705, 'scatthered': 961, 'ringed': 4707, 'sings': 977, 'angle': 4709, 'catus': 211, 'birn': 4711, 'ashort': 4712, 'parrots': 4713, 'bick': 4714, 'bough': 4715, 'physique': 1886, 'nosed': 1675, 'eyerbow': 4718, 'smallbill': 1033, 'roun': 4720, 'mix': 4721, 'sprinkled': 4439, 'coverts': 4113, 'astride': 1368, 'which': 4724, 'tinged': 5334, 'relativey': 3226, 'jail': 4387, 'target': 4728, 'blades': 1096, 'patchy': 4730, 'ray': 3721, 'sm': 4732, 'clash': 4733, 'hazel': 1459, 'torse': 4737, 'knot': 5027, 'patchj': 4739, 'winglet': 4740, 'eight': 1179, 'bel': 4663, 'cracking': 4743, 'peppered': 4744, 'auklet': 3776, 'draws': 2104, 'thebreast': 4746, 'boundary': 4747, 'patche': 4748, 'fair': 4749, 'dorsal': 4750, 'death': 3571, 'sheathed': 2125, 'toned': 2625, 'placement': 4753, 'wite': 4754, 'malor': 1262, 'flappers': 4756, 'blackbeak': 4757, 'spiculated': 4471, 'wight': 1967, 'ringeye': 1286, 'face': 4761, 'looked': 4762, 'tapers': 5342, 'ligther': 3444, 'brey': 4765, 'distinctive': 3546, 'speckles': 3486, 'built': 4767, 'coverering': 1334, 'skunk': 4769, 'normally': 4770, 'inand': 4771, 'purples': 4772, '50': 4773, 'warblers': 4423, 'colorations': 4777, 'spindly': 4778, 'larsud': 4157, 'diluted': 4780, 'feathets': 4781, 'supported': 33, 'woolly': 4783, 'brewers': 4784, 'beally': 4231, 'hooklike': 4785, 'connect': 4786, 'bring': 3242, 'senegal': 4788, 'coevering': 4789, 'slapped': 4791, 'greycolor': 3337, 'chicken': 2946, 'darkest': 4795, 'eyes': 1910, 'seagul': 947, 'ruddy': 4798, 'glows': 4118, 'creamed': 1480, 'coloring': 1485, 'based': 4802, 'earthtone': 4803, 'jay': 4804, 'eyebroews': 1806, 'jaw': 4806, 'superciliari': 4807, 'fleshy': 5277, 'songbird': 3553, 'superceliary': 4809, 'bug': 5347, 'thrat': 4811, 'surroundings': 1541, 'lightest': 4180, 'tape': 1555, 'shinny': 4814, 'wkth': 4815, 'breasting': 4221, 'hads': 1577, 'b': 3482, 'staulky': 3163, 'threateningly': 4820, 'meant': 4161, 'expandable': 4822, 'haircut': 4823, 'agrey': 220, 'beat': 4825, 'symetrical': 1639, 'aalso': 4827, 'abdoomen': 4828, 'overall': 4829, 'stripes': 4830, 'bear': 4831, 'whoe': 4832, 'beal': 4833, 'joint': 1668, 'bean': 4835, 'whitebelly': 104, 'winglays': 4837, 'bulbously': 4838, 'probably': 3557, 'blaze': 4128, 'windspan': 832, 'soiled': 3659, 'thrives': 1969, 'squatty': 3753, 'areas': 4844, 'wee': 4845, 'gray': 4846, 'bellies': 3558, 'places': 4847, 'blacj': 2075, 'numerous': 4398, 'eyebros': 4852, 'shorebird': 3085, 'comprising': 4853, 'eyebrow': 4854, 'tuned': 1793, 'tappers': 4856, 'topaz': 4857, 'nutmeg': 4858, 'pictured': 3965, 'gran': 4860, 'contain': 4861, 'thraot': 4862, 'ohio': 4863, 'grab': 4864, 'predatory': 4865, 'looks': 4451, 'monochromatic': 1859, 'ofcblack': 4868, 'scalloped': 5353, 'variations': 5357, 'puddle': 1905, 'birdy': 3433, 'shrouded': 4872, 'packet': 4873, 'intensity': 4874, 'stirpe': 4875, 'slanted': 4876, 'tricolored': 1945, 'spotter': 597, 'violet': 4879, 'overtones': 4880, 'luminous': 4882, 'woodland': 4883, 'stop': 4410, 'balc': 4886, 'awaits': 3689, 'closer': 4888, 'wire': 4890, 'underparts': 807, 'crwon': 4507, 'coveting': 4136, 'posing': 4893, 'beading': 2038, 'segment': 4895, 'below': 5360, 'clor': 5121, 'musky': 4897, 'state': 4898, 'proof': 4899, 'patters': 4900, 'closed': 4901, 'masking': 4902, 'purgy': 4903, 'stories': 4904, 'stips': 3778, 'bunch': 4906, 'complement': 4907, 'mainly': 1694, 'dirty': 2635, 'perfect': 4457, 'comparable': 2129, 'nashville': 4912, 'pronged': 4913, 'whitecovering': 3333, 'spares': 4915, 'horizontally': 4850, 'joy': 4917, 'limbs': 1711, 'vectrices': 4919, 'day': 5142, 'lonjg': 4920, 'hooks': 4105, 'ducklike': 2204, 'blackbird': 2638, 'disproportionately': 2192, 'configuration': 4925, 'swift': 4926, 'regular': 3917, 'redheaded': 2676, '<end>': 0, 'centric': 4930, 'pepper': 4181, 'taking': 4933, 'exclusing': 4934, 'broadside': 1583, 'equal': 4936, 'display': 4937, 'etc': 4938, 'coverlet': 4240, 'streams': 2325, 'flatt': 4940, 'slitted': 4941, 'bronw': 4942, 'section': 2845, 'puff': 4944, 'beak': 4839, 'brosd': 5371, 'glorious': 4946, 'otherwise': 4947, 'detailed': 2643, 'coverets': 3837, 'loon': 4950, 'co': 4951, 'triangular': 5337, 'throatted': 4333, 'dosile': 4952, 'encompasses': 4953, 'denim': 4954, 'overcoat': 4955, 'penetrating': 5373, 'environments': 2436, 'high': 4372, 'visually': 4958, 'orangs': 4465, 'andwhite': 4960, 'polka': 4974, 'russet': 2647, 'narrowed': 4962, 'ccrown': 5078, 'iwngs': 4963, 'mourning': 4964, 'chickadee': 4965, 'flashes': 4677, 'outsides': 4966, 'catches': 4967, 'primaires': 3714, 'lw': 4675, 'tremendous': 2541, 'breastts': 3576, 'muddy': 4972, 'yellowgray': 4401, 'allover': 1530, 'dazzling': 4975, 'colorul': 3577, 'immense': 4977, 'gaze': 4978, 'slowly': 4979, 'fuzzy': 4655, 'certain': 4135, 'fiacial': 1184, 'bluegreen': 4982, 'feather': 4983, 'feathes': 280, 'deet': 4986, 'fence': 2817, 'uplifted': 1918, 'bras': 4669, 'shoulders': 4990, 'browner': 4991, 'corresponding': 3230, 'consistant': 5114, 'painted': 4993, 'intelligent': 4994, 'al': 2654, 'bulk': 4996, 'watches': 4997, 'calmly': 2562, 'covered': 4473, 'adding': 4959, 'nck': 3419, 'ots': 2749, 'bule': 2759, 'wingfeathers': 3583, 'encircles': 1916, 'eyelashes': 1588, 'wins': 2833, 'couple': 3813, 'splits': 1475, 'unlike': 5009, 'bellyq': 5010, 'edging': 4474, 'vanilla': 5012, 'fisheyed': 3881, 'harder': 2819, 'bellyy': 2834, 'larege': 2837, 'averagely': 5018, 'hovering': 569, 'accentuates': 52, 'plateaued': 5021, 'encircled': 2866, 'cheekspot': 4345, 'notiable': 2872, 'primaily': 5025, 'bellyl': 5026, 'ly': 4476, 'ombre': 2891, 'bray': 4881, 'carved': 5029, 'swimmer': 4969, 'taupe': 2915, 'eyeball': 5032, 'eblack': 2922, 've': 5034, 'ripe': 5035, 'geometric': 5036, 'almost': 5037, 'wiwth': 5038, 'againsst': 3055, 'coating': 4469, 'surprisingly': 1939, 'surface': 2034, 'ostly': 5043, 'bead': 4840, 'supercallirary': 5044, 'remiges': 2994, 'mockingbird': 5046, 'dual': 5047, 'sheath': 5048, 'partner': 5049, 'sits': 5050, 'sitt': 3028, 'perrywinkle': 5052, 'whitefeathers': 5053, 'greater': 4480, 'pied': 5163, 'ehad': 5055, 'highlighter': 5056, 'whire': 1065, 'pit': 5058, 'balloon': 5059, 'progressing': 5060, 'lousiana': 379, 'eeybrow': 2618, 'himself': 4238, 'guards': 2992, 'when': 2377, 'strange': 3126, 'makings': 4848, 'inch': 4012, 'cream': 2670, 'pengun': 5020, 'outlined': 4485, 'flair': 5279, 'difficult': 5069, 'elbows': 5071, 'ball': 5072, 'balk': 5073, 'stoutly': 4910, 'complemented': 5075, 'smaill': 3205, 'eccentric': 5077, 'gorgeous': 2944, 'bald': 5079, 'tints': 1935, 'effect': 5081, 'coffe': 5082, 'scratched': 4011, 'colouring': 5084, 'tufts': 3747, 'robotic': 3263, 'fierce': 3268, 'arrowhead': 5087, 'trasus': 5088, 'collar': 5089, 'mosaic': 5090, 'stunningly': 5091, 'staright': 5092, 'off': 5093, 'reflection': 3322, 'rectricles': 5095, 'neural': 5096, 'i': 5097, 'downwward': 5098, 'colour': 5099, 'variety': 5399, 'banded': 3349, 'underbill': 3049, 'yelllow': 4608, 'scissor': 3368, 'staight': 5105, '1': 4458, 'flaired': 4949, 'colod': 5107, 'sets': 3393, 'markings': 1746, 'butterball': 5110, 'drawing': 5111, 'itschest': 5112, 'usual': 4574, 'spherical': 3429, 'dail': 3437, 'stout': 4494, 'attractively': 5118, 'less': 306, 'oak': 5402, 'wren': 3464, 'tailfeathering': 2820, 'submerges': 4242, 'gulls': 5124, 'detail': 5403, 'tursus': 5126, 'underlying': 5127, 'distant': 5128, 'fishing': 5129, 'cammoflage': 5130, 'miniscule': 5131, 'tough': 5132, 'cartoons': 3505, 'ashy': 5273, 'brilliantly': 5305, 'web': 5133, 'feathered': 1751, 'likes': 3610, 'ther': 5135, 'exquisite': 4841, 'field': 5137, 'enlongated': 5138, 'lakc': 3598, 'sky': 5140, 'jas': 4808, 'bthis': 660, 'arresting': 5143, 'browninsh': 5144, 'add': 5145, 'book': 5146, 'combine': 1193, 'attractive': 5148, 'tawny': 5149, 'wet': 5150, 'adn': 5151, 'stable': 4130, 'bird1': 2475, 'ski': 5154, 'hewn': 5155, 'match': 5156, 'raven': 5157, 'knob': 574, 'sotted': 5159, 'definitive': 5160, 'layering': 5161, 'tiped': 4702, 'spckles': 385, 'bowling': 5243, 'thicker': 4797, 'crushed': 5165, 'semicircled': 5166, 'increases': 5167, 'frayed': 5168, 'five': 5169, 'fantail': 1191, 'zigzags': 5289, 'billl': 5171, 'contrasting': 3887, 'bellyand': 2486, 'pies': 4312, 'khaki': 2386, 'name': 1759, 'newborn': 5175, 'crisp': 3552, 'like': 5177, 'lost': 4921, 'brownbody': 850, 'bulge': 5180, 'alternatively': 3841, 'inalar': 2718, 'grean': 4726, 'candy': 3863, 'units': 5067, 'sized': 5185, 'colroing': 5186, 'roundly': 5187, 'straihgt': 5188, 'hosts': 5189, 'heed': 5190, 'martin': 5191, 'wblack': 5192, 'blakish': 1887, 'clinging': 5194, 'soft': 5195, 'radiant': 5004, 'isbonze': 853, 'wiskers': 2792, 'because': 5199, 'glare': 5200, 'shabby': 5201, 'habitat': 3605, 'sequence': 2032, 'concave': 5204, 'portly': 5205, 'meedium': 5206, 'armoring': 3972, 'meld': 5207, 'hair': 3990, 'tucted': 3994, 'aligned': 5210, 'growth': 5211, 'brelly': 5212, 'flush': 497, 'ablaze': 3071, 'home': 5214, 'mustard': 4059, 'throughout': 3613, 'highlighting': 4076, 'tuffled': 4973, 'lead': 5219, 'overlay': 1610, 'dunes': 5221, 'broad': 5222, 'leaving': 5223, 'contently': 4100, 'lean': 5225, 'rattlesnake': 5226, 'eyepatches': 3614, 'rumpled': 5228, 'does': 5230, 'masked': 5231, 'intimidating': 5377, 'ripped': 5232, 'enourmous': 5233, 'backend': 5234, 'vlue': 5235, 'underwing': 4167, 'pasty': 5237, 'rumpp': 5238, 'thinly': 3417, 'nourishment': 5421, 'coated': 5240, 'dard': 3803, 'reaching': 4792, 'midair': 5242, 'spckled': 5173, 'triceps': 4317, 'overbite': 5244, 'sizes': 5179, 'mettalic': 3413, 'tucks': 5247, 'showcasing': 1772, 'host': 5249, 'outrageously': 5250, 'although': 5251, 'accompanied': 4519, 'fkat': 5253, 'fluorescent': 5254, 'sceondary': 215, 'sways': 4099, 'samall': 4146, 'iris': 5258, 'about': 5259, 'bills': 4521, 'rare': 4327, 'promement': 5262, 'getting': 5263, 'traces': 4522, 'ribbed': 5265, 'featjhers': 5266, 'shipping': 2710, 'underfeathers': 5268, 'blush': 5269, 'union': 5270, 'carries': 5271, 'seeds': 5272, 'blackness': 293, 'beaked': 5274, 'tailed': 5275, 'doted': 2821, 'poof': 4843, 'tongue': 5278, 'upward': 4523, 'eloquently': 2571, 'tome': 5281, 'decurved': 2842, 'tent': 5283, 'fethered': 4443, 'reptile': 4447, 'own': 4449, 'midsize': 2702, 'yhellow': 5288, 'prominently': 3648, 'comicall': 2371, 'atil': 5290, 'spottedbelly': 5291, 'pearlish': 5292, 'outward': 4719, 'dry': 5294, 'guard': 5295, 'en': 5296, 'recitces': 5297, 'erect': 942, 'female': 5299, 'lardge': 2705, 'kinda': 5301, 'stripey': 4824, 'crooked': 2146, 'hang': 5302, 'decorated': 2706, 'retrics': 3280, 'significantly': 1779, 'ting': 2341, 'downwad': 5307, 'openings': 5308, 'additional': 5042, 'confident': 5310, 'nbelly': 5311, 'flustered': 5313, 'outline': 4659, 'rentrices': 5314, 'andhas': 2675, 'curvbed': 1169, 'brast': 3996, 'tails': 5109, 'much': 877, 'browish': 5321, 'inner': 5322, 'posterior': 4759, 'biill': 4528, 'powdered': 5324, 'extravagant': 3628, 'mallar': 2398, 'grenade': 5327, 'coarse': 3629, 'withgreen': 2709, 'mailer': 969, 'crushing': 5331, 'swimmimg': 5332, 'remaining': 3805, 'hs': 4725, 'beaming': 5336, 'ead': 4738, 'but': 5338, 'waddler': 5339, 'reminds': 1775, 'repeated': 1119, 'lowered': 4763, 'nut': 4638, 'protrusions': 4194, 'brighly': 188, 'ear': 1130, 'eat': 4810, 'tinges': 5348, 'partially': 4156, 'also': 5350, 'made': 5351, 'unproportional': 2658, 'barrow': 5153, 'wish': 3638, 'heavyset': 5355, 'georgous': 4045, 'smooth': 4869, 'iscovered': 5358, 'placed': 5359, 'carriage': 4896, 'graphite': 3358, 'cerulean': 5364, 'throughtout': 5365, 'demonstrate': 5366, 'diagonally': 5367, 'piece': 5368, 'iridiscent': 3971, 'birid': 3633, 'russian': 5370, 'averaged': 4344, 'dowwnward': 4678, 'slips': 3636, 'enhanced': 4878, 'secndaries': 5374, 'entering': 5102, 'pin': 5376, 'resting': 3637, 'mutli': 892, 'pic': 5379, 'buffy': 5380, 'atop': 3976, 'strpe': 5442, 'standing': 4284, 'firey': 3285, 'distinguished': 4681, 'vines': 5385, 'highlighing': 5039, 'beaty': 5387, 'partly': 5388, 'reflecting': 5389, 'ing': 5390, 'leadingto': 1915, 'margins': 5392, 'primairies': 5393, 'redder': 5070, 'tick': 5172, 'compared': 5396, 'woodpeckers': 5397, 'loud': 5398, 'incredible': 5100, 'yellowbreast': 5400, 'bulky': 4603, '45': 5401, 'percent': 5120, 'forests': 5125, 'cheeckpatch': 1363, 'blown': 4606, 'smears': 5406, 'mast': 1182, 'uni': 5408, 'special': 4542, 'branch': 5410, 'finch': 5411, 'rimmed': 2722, 'incredibly': 5413, 'stumpy': 347, 'illusion': 5415, 'outlines': 4499, 'sloping': 5416, 'agrayish': 5417, 'uppertail': 4090, 'cameraperson': 5419, 'living': 5420, 'stringy': 3507, 'secondarines': 5422, 'sustains': 5423, 'variation': 5424, 'stay': 2086, 'noticed': 5317, 'rey': 5426, 'rotund': 5427, 'cheekstrip': 2107, 'wind': 2027, 'seismic': 5430, 'curbed': 5431, 'coloration': 5432, 'upwards': 5433, 'chalky': 5329, 'droplet': 5435, 'breaks': 5436, '4x': 5437, 'described': 5438, 'tanned': 5439, 'littered': 4775, 'jewel': 5441, 'arrowpoints': 5382, 'biled': 5443, 'strings': 5444, 'midszie': 4689, 'chippy': 5445, 'portion': 5446, 'pupil': 5447, 'yell': 5448, 'oversize': 5449}
  print("||||||||||||||||||||||||||||||||||| WORD TO INDEX ||||||||||||||||||||||||||||||||||||||||")  
  n_words = len(wordtoix)
  filepath = 'example_filenames.txt' #a group of example filenames, 24 names for 24 file, each file with a number of captions
  data_dic = {}  # dictionary used to generate images from captions

  with open(filepath, "r") as f:
      filenames = f.read().split('\n')
      #print("2.Opened and loaded example_filename.txt")
      first_counter = 0
      for name in filenames:
          first_counter += 1
          if name == "example_captions":  #Keep this way until you download the rest of the file captions
              filepath = '%s.txt' % (name)
              with open(filepath, "r") as f:
                  sentences = f.read().split('\n')
                  #print("3.Opened and loaded example_captions.txt(",first_counter,") in loop of all names of filename")
                  # split your text file of 16 captions to a list of 16 string entries
                  captions = []
                  cap_lens = []
                  second_counter = 0
                  for sent in sentences:
                      second_counter += 1
                      sent = sent.replace("\ufffd\ufffd", " ")
                      tokenizer = RegexpTokenizer(r'\w+')
                      tokens = tokenizer.tokenize(sent.lower())
                      #convert a single sentence(string) to list of tokens(words)=>result in a list of string entries that are word that make up the original sentence(caption)
                      if second_counter == 0 :
                        second_counter = 0
                        #print("4.Tokenized a sentence from example file in filenames (", second_counter,")" )
                      rev = []
                      third_counter = 0
                      for t in tokens:
                          third_counter += 1
                          t = t.encode('ascii', 'ignore').decode('ascii')
                          rev.append(wordtoix[t])
                          if third_counter == 0 :
                            third_counter = 0
                            #print ("5.append converted token to new list(",third_counter,")")
                              #convert the list of words to a list crosspending indexes
                      captions.append(rev)  # all captions in the file
                      cap_lens.append(len(rev))
                      if second_counter == 0:
                        second_counter = 0
                        #print("6.append converted sentence to list of captions (",second_counter,")")
                      # the length(number of words/tokens) in each caption
              max_len = np.max(cap_lens)  # used to pad shorter captions
              sorted_indices = np.argsort(cap_lens)[::-1] # Returns the indices that would sort the array of lengths.
              cap_lens = np.asarray(cap_lens)
              cap_lens = cap_lens[sorted_indices]# sort the array of lengths using the sotring indices
              cap_array = np.zeros((len(captions), max_len), dtype='int64')  #placeholder for the padded sorted array caption
              for i in range(len(captions)):
                  idx = sorted_indices[i]
                  cap = captions[idx]
                  c_len = len(cap)
                  cap_array[i, :c_len] = cap
              #print("7.created padded sorted array of caption cap_array (", first_counter,")")
              key = name[(name.rfind('/') + 1):]
              data_dic[key] = [cap_array, cap_lens, sorted_indices]
              #print("8. created data_dic of cap_array , cap_lens & sorted_indices (", first_counter,")")


  #print("9. Calling gen_example(data_dic) ")
  f_images = gen_example(data_dic ,n_words )
  return f_images



def discriminator_loss(netD, fake_imgs):
    fake_labels = Variable(torch.FloatTensor(Global_Batch_size).fill_(0))
    print("fake_labels", fake_labels)
    print(fake_imgs.shape)
    print('-------------------------')
    fake_features = netD(fake_imgs)
    print("fake_features", fake_features)
    if netD.UNCOND_DNET is not None:
        fake_logits = netD.UNCOND_DNET(fake_features)
        print("fake_logits",fake_logits)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        print("fake_errD", fake_errD)
        errD = fake_errD
    return errD




def GenerateImages(Input_Caption ="This bird has red wings white belly black head ", bs = 1, custome_input = False):
    CurrentLoss = 0
    while CurrentLoss < 1.5 :


        currentDirectory = os.getcwd()
        os.chdir('./content/Bird-Image-Generator')

        if custome_input:
            with open("example_captions.txt", 'w') as f:
                f.write(Input_Caption)

        if not custome_input:
            samples = [
                'this bird is red with white and has a very short beak',
                'the bird has a yellow crown and a black eyering that is round',
                'this bird has a green crown black primaries and a white belly',
                'this bird has wings that are black and has a white belly',
                'this bird has wings that are red and has a yellow belly',
                'this bird has wings that are blue and has a red belly',
                'this is a small light gray bird with a small head and green crown, nape and some green coloring on its wings',
                'his bird is black with white and has a very short beak',
                'this small bird has a deep blue crown, back and rump and a bright white belly',
                'this is a blue bird with a white throat, breast, belly and abdomen and a small black pointed beak',
                'yellow abdomen with a black eye without an eye ring and beak quite short',
                'this bird is yellow with black on its head and has a very short beak',
                'this bird has wings that are black and has a white belly',
                'this bird has a white belly and breast with a short pointy bill',
                'this bird is red and white in color with a stubby beak and red eye rings',
                'a small red and white bird with a small curved beak'
            ]
            from random import randrange
            index = (randrange(len(samples)))

            Input_Caption = samples[index]

            with open("example_captions.txt", 'w') as f:
                f.write(Input_Caption)
            bs = 1

        fake_images = the_main()

        netD = D_NET256()
        model_dir = NET_D
        state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
        netD.load_state_dict(state_dict)
        netD.eval()

        xx = display_images(bs, currentDirectory)
        print(xx)

        dloss = discriminator_loss(netD, fake_images[2])

        print('The Loss', dloss)


        CurrentLoss = dloss.item()

    return Input_Caption


