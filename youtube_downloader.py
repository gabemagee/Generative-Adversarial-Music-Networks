import os
import time

import pytube
from pytube import YouTube
import subprocess

# where to save

# link of the video to be downloaded


def download_audio_from_list(urls,dest=None):
    if dest!=None:
        os.chdir(dest)
    for url in urls:
        print(url)
        try:
            yt = YouTube(url)
            stream_a = yt.streams.filter(progressive=True).get_by_itag(22)
            stream_b = yt.streams.filter(progressive=True).get_by_itag(18)


            t1 = time.time()
            print(stream_a)
            s = stream_a.download(filename=url+"_a")
            s = s.split("\\")[-1]
            mp4_to_mp3(s)
            t2 = time.time()
            t3 = time.time()
            print(t2-t1)
            print(stream_b)
            s = stream_b.download(filename=url + "_b")
            s = s.split("\\")[-1]
            mp4_to_mp3(s)
            t4 = time.time()
            print(t4-t3)

        except pytube.exceptions.RegexMatchError:
            print("May be copyrighted")


def mp4_to_mp3(filename):
    dest = filename.split(".")[0]+".mp3"
    if os.path.exists(dest):
        os.remove(dest)
    cmd = "ffmpeg -loglevel panic -i "+filename+" "+dest
    subprocess.check_output(cmd)
    os.remove(filename)
    return dest


f = open("yt.txt","r")
download_audio_from_list(f.readlines())
f.close()
