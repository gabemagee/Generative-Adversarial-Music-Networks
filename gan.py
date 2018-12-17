import random
#import torchaudio
import os
#import torch
#from pathlib import Path

def n_random_elts(lst,n):
    sample = []
    original = lst[:]
    for k in range(n):
        i = random.randint(0,len(original)-1)
        selection = original[i]
        sample.append(selection)
        del original[i]
    return sample, original

def testing_and_validation(directory,ratio):
    files = list(os.listdir(directory))
    for i in range(len(files)):
        files[i] = directory+"/"+files[i]
    n = len(files)
    size = int(n*ratio)
    testing_set, validation_set = n_random_elts(files,size)
    return testing_set,validation_set

def zip_tuple(lst,value):
    ret = []
    for item in lst:
        ret.append((item,value))
    return ret

def main():
    #divide samples in training and validation set
    directory = os.getcwd()
    ratio = 0.7
    assert('post' in list(os.listdir(directory)))
    os.chdir(directory+"/post")
    folders = os.listdir(os.getcwd())
    assert('positive' in folders)
    assert('negative' in folders)
    p = os.getcwd()+'/positive'
    n = os.getcwd()+'/negative'
    t_p, v_p = testing_and_validation(p,ratio)
    t_n, v_n = testing_and_validation(n,ratio)
    testing_set = zip_tuple(t_p,1) + zip_tuple(t_n,0)
    validation_set = zip_tuple(v_p,1) + zip_tuple(v_n,0)
    #train discriminator
    #validation set test
    #GAN cycle
        #train forger for x iterations - save data
        #train discriminator for x iterations on data
        #produce wav files

if __name__ == '__main__':
    main()
