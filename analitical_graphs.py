import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import matplotlib
import threading


def main(file):
    p=signal(file)
    j=spectrum(file)
    return (p and j)

def signal(file):
    y, sr = librosa.load(file)
    plt.plot(y)
    plt.title('Signal of voice sample')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    #filepath = os.path.join(os.getcwd(), 'images/graphs/Signal.png')
    
    plt.savefig('images/graphs/Signal.png',dpi=50)
    plt.cla()
    return True

def spectrum(file): 
    y, sr = librosa.load(file)   
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
    plt.plot(ft)
    plt.title('Spectrum')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    
    #filepath = os.path.join(os.getcwd(), 'images/graphs/Spectrum.png')
    plt.savefig('images/graphs/Spectrum.png',dpi=50)
    return True
    """spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    filepath = os.path.join(os.getcwd(), 'images/graphs/Spectrogram.png')
    plt.savefig(filepath)"""

#file="test-samples/27-124992-0002.wav"
#j=main(file)


    