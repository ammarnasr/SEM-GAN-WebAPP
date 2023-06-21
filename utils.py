import os
from shutil import copy


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

    copy(filename2, dst)
    fileNamesList.append(filename2)
    # display(z)

  os.chdir(currentDirectory)
  return fileNamesList


def mkdir_p(path):
  """make dir if not exist"""
  if not os.path.exists(path):
    os.makedirs(path)