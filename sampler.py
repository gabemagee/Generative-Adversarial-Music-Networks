import os
import subprocess
import time
import wave


def get_length(filename):
    f =wave.openfp(filename,"r")
    n =f.getnframes()
    fr = f.getframerate()
    l = n/fr
    return l

def create_samples(seconds,framerate):
    t0 = time.time()
    label = {1:"positive",0:"negative"}
    c =0
    for i in [0,1]:
        print("Creating "+label[i]+" examples")
        o = os.getcwd()
        src = o+"\\pre\\"+label[i]
        dest = o+"\\post\\"+label[i]
        os.chdir(src)
        folders = [str(i[0]) for i in os.walk(os.getcwd())]
        for folder in folders:
            os.chdir(folder)
            files = os.listdir(os.getcwd())
            for file in files:
                if os.path.isfile(file):
                    file = to_wav(file)
                    c = file_to_samples(file,dest,seconds,framerate,c)+1
        os.chdir(o)
    t = time.time() - t0
    print(t)
    print(c)
    if c!=0:
        print(t/c)

def to_wav(infile):
    if infile[-3:] == "jpg":
        return None
    elif infile[-3:]!="wav":
        outfile = "\""+infile[:-4] + ".wav\""
        infile = "\""+infile+"\""
        cmd = "ffmpeg  -loglevel panic -i "+infile+" "+outfile
        subprocess.check_output(cmd)
        os.remove(infile[1:-1])
        outfile = outfile[1:-1]
        return outfile
    else:
        return infile

def compress(input, output, rate):
    command = "ffmpeg -loglevel panic -i \""+input +"\" -ar " + str(rate) + " \""+output+"\""
    return subprocess.check_output(command)

def snip(input,output,start, duration):
    end = str(start+duration)
    start = str(start)
    command = "ffmpeg -loglevel panic -i "+"\""+input+"\" -ss "+start+" -to "+end+" -c copy \""+output+"\""
    return subprocess.check_output(command)

def to_mono(input,output):
    command = "ffmpeg -loglevel panic -i \""+input+"\" -ac 1 \""+output+"\""
    return subprocess.check_output(command)

def make_sample(input,dir,iter,start,duration,rate):
    placeholder = "pl.wav"
    placeholder2 = "pl2.wav"
    dest = str(iter) + ".wav"
    try:
        os.remove(placeholder)
    except:
        pass
    try:
        os.remove(placeholder2)
    except:
        pass
    try:
        os.remove(dest)
    except:
        pass
    try:
        os.remove(dir+"\\"+dest)
    except:
        pass
    snip(input,placeholder,start,duration)
    compress(placeholder,placeholder2,rate)
    to_mono(placeholder2, dest)
    os.remove(placeholder)
    os.remove(placeholder2)
    os.rename(dest,dir+"\\"+dest)




def main():
    #make_sample("HEAT.wav","positive_samples_post",0,0,15,10000)
    create_samples(15,10000)

def file_to_samples(file,dir,per,framerate,count):
    r = count
    if file!=None:
        l = get_length(file)
        chunks = int(l / per)
        for i in range(chunks):
            start = i * per
            r = i + count
            make_sample(file, dir, r, start, per, framerate)
        return r
    else:
        return r-1

if __name__ == '__main__':
    main()