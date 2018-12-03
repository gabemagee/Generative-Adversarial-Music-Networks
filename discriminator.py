import torchaudio

audio_tensor, sample_rate = torchaudio.load('HEAT.wav')
sz = audio_tensor.size()
print(sz,sample_rate)
