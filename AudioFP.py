import numpy as np
import pydub
import scipy.signal 
import skimage.feature 
import datasketch 
from bokeh.plotting import figure, show
import pyaudio
import warnings
import sys
import pickle 
import os
# import pyhash

# Parameters for tuning the Audiofingerprinting algorithm

# Parameters used in generating spectrogram
#----------------------------------
nperseg = 16 * 256  # window size
overlap_ratio = 0.4  # degree of overlap, larger number->more overlap, denser fingerprint
#----------------------------------

# Parameters used in finding local peaks
#-------------------------
min_peak_sep = 15  # larger sep -> less peaks -> less accuracy, but faster fingerprinting
min_peak_amp = 10  # larger min amp -> less peaks -> less accuracy, but faster fingerprinting
#-------------------------

# Parameters used in generating fingerprint
#------------------------------
peak_connectivity = 15  # Number of neighboring peaks to use as target for each anchor
peak_time_delta_min = 0  # Minimum spacing in time between peaks for anchor and target
peak_time_delta_max = 200  # Maximum spacing in time between peaks for anchor and target
#------------------------------

# This class defines an AudioFP object that stores the name a song and its fingerprint. 
# It also contains all the functions used to read/record audio, generate spectrogram, find peaks,
# generate fingerprint, and saving the object to file.

class AudioFP():
    
    # Initializing AudioFP object.
    
    # Creating the AudioFP class object will prompt the user to chose whether they would like to read audio
    # from a file or to record using a microphone, or to open an already saved object. 
    # Enter 'f' to read an audio file, 'r' to record, or 's' to open a saved object. 
    # Entering any other character will cause the program to throw an exception and exit.
    # The user is also prompted to choose whether they want to generate plots. 
    # Enter 'y' to generate plots or 'n' to skip plotting.
    # After these user selections are made, the program automatically reads/records audio, generate
    # a sprectrogram, finds the local peaks in the spectrogram, and generates a fingerprint 
    # or simply reads an existing AudioFP object from file.
    # Finally, if the user chose to read audio from file or record it, they are prompted to choose
    # whether they want to save the object to file. Enter 'y' to save or 'n' to skip.

    def __init__(self, process='m'):
        self.songname = ''
        # hasher = pyhash.farm_32()
        # self.fingerprint = datasketch.MinHash(num_perm=512, hashfunc=hasher)
        self.fingerprint = datasketch.MinHash(num_perm=512)
        self.framerate = []
        if process == 'a':
            self.ask_user()
        elif process == 'm':
            pass
        else:
            if input('Enter "a" for automated fingerprinting or "m" to proceed manually: ') == 'a':
                self.ask_user()
            else:
                sys.exit('''Error: Incorrect entry.''')
        
    def ask_user(self):
        audio_type = input('Enter "f" to read from audio file or "s" to open saved fingerprint: ')
        if audio_type == 'f':
            filename = input('Enter the filename you want to read (excluding the extension): ')
            self.songname = filename
            if input('Do you want to show all plots? Enter "y" or "n": ') == 'y':
                plot = True
            else:
                plot = False
            channels, self.framerate = self.read_audiofile(plot, filename)
            f, t, sgram = self.generate_spectrogram(plot, channels, self.framerate)
            fp, tp, peaks = self.find_peaks(plot, f, t, sgram)
            self.generate_fingerprint(plot, fp, tp, peaks)
            if input('Do you want to save the fingerprint to file for later use? Enter "y" or "n": ') == 'y':
                self.save_fingerprint()
            else:
                print('Not saving anything')
        elif audio_type == 's':
            objname = input('Enter the filename (excluding the extention) where the fingerprint is saved: ')
            objname = os.getcwd() + '/audio-fingerprinting/songs/' + objname + '.pkl'
            with open(objname, 'rb') as inputobj:
                data = pickle.load(inputobj)
                self.songname = data['songname']
                self.fingerprint = data['fingerprint']
                self.framerate = data['framerate']
            if input('Do you want to see the details of the file? Enter "y" or "n": ') == 'y':
                plot = True
                print('Songname: ', self.songname)
                print('Framerate: ', self.framerate)
                print('Audio-fingerprint:')
                print(self.fingerprint.digest())
            else:
                plot = False
        else:
            sys.exit('''Error: Incorrect entry. Enter "f" to read an audio file, 
                     "r" to record, or "s" to open a saved object''')
        
    # Read audio file using pydub and plot signal.
    # The audio file has to be .mp3 format
    def read_audiofile(self, plot, filename):
        songdata = []  # Empty list for holding audio data
        channels = []  # Empty list to hold data from separate channels
        filename = os.getcwd() + '/audio-fingerprinting/songs/' + filename
        audiofile = pydub.AudioSegment.from_file(filename + '.mp3')
        audiofile = audiofile[:30000].fade_out(2000)  # keeping only first 20s and adding fade out (to reduce RAM requirements)
        self.songname = os.path.split(filename)[1]
        songdata = np.frombuffer(audiofile._data, np.int16)
        for chn in range(audiofile.channels):
            channels.append(songdata[chn::audiofile.channels])  # separate signal from channels
        framerate = audiofile.frame_rate
        channels = np.sum(channels, axis=0) / len(channels)  # Averaging signal over all channels
#         channels = channels[:int(2 * len(channels) / 3)]
        # Plot time signal
        if plot:
            p1 = figure(plot_width=900, plot_height=500, title='Audio Signal', 
                        x_axis_label='Time (s)', y_axis_label='Amplitude (arb. units)')
            time = np.linspace(0, len(channels)/framerate, len(channels))
            p1.line(time[0::1000], channels[0::1000])
            show(p1)
        return channels, framerate
        
    # Generate and plot spectrogram of audio data
    def generate_spectrogram(self, plot, audiosignal, framerate):
        fs = framerate  # sampling rate
        window = 'hamming'  # window function
        noverlap = int(overlap_ratio * nperseg)  # number of points to overlap
        # generate spectrogram from consecutive FFTs over the defined window
        f, t, sgram = scipy.signal.spectrogram(audiosignal, fs, window, nperseg, noverlap)  
        sgram = 10 * np.log10(sgram)  # transmorm linear output to dB scale 
        sgram[sgram == -np.inf] = 0  # replace infs with zeros
        # Plot Spectrogram
        if plot:
            p2 = figure(plot_width=900, plot_height=500, title='Spectrogram',
                        x_axis_label='Time (s)', y_axis_label='Frequency (Hz)',
                        x_range=(min(t), max(t)), y_range=(min(f), max(f)))
            p2.image([sgram[::2, ::2]], x=min(t), y=min(f), 
                     dw=max(t), dh=max(f), palette='Spectral11')
            show(p2)
        return f, t, sgram
        
    # Find peaks in the spectrogram using image processing
    def find_peaks(self, plot, f, t, sgram):
        coordinates = skimage.feature.peak_local_max(sgram, min_distance=min_peak_sep, indices=True,
                                     threshold_abs=min_peak_amp)
        
        peaks = sgram[coordinates[:, 0], coordinates[:, 1]]
        tp = t[coordinates[:, 1]]
        fp = f[coordinates[:, 0]]
        # Plot the peaks detected on the spectrogram
        if plot:
            p3 = figure(plot_width=900, plot_height=500, title='Spectrogram with Peaks',
                        x_axis_label='Time (s)', y_axis_label='Frequency (Hz)',
                        x_range=(min(t), max(t)), y_range=(min(f), max(f)))
            p3.image([sgram[::2, ::2]], x=min(t), y=min(f), 
                     dw=max(t), dh=max(f), palette='Spectral11')
            p3.scatter(tp, fp)
            show(p3)
        return fp, tp, peaks
        
    # Use the peak data from the spectrogram to generate a string with pairs of 
    # peak frequencies and the time delta between them.
    def generate_fingerprint(self, plot, fp, tp, peaks):
        # Create the data to be used for fingerprinting
        # for each frequency (anchor) find the next few frequencies (targets) and calculate their time deltas
        # the anchor-target frequency pairs and their time deltas will be used to generate the fingerprints
        s = []  # Empty list to contain data for fingerprint
        for i in range(len(peaks)):
            for j in range(1, peak_connectivity):
                if (i + j) < len(peaks):
                    f1 = fp[i]
                    f2 = fp[i + j]
                    t1 = tp[i]
                    t2 = tp[i + j]
                    t_delta = t2 - t1
                    if t_delta >= peak_time_delta_min and t_delta <= peak_time_delta_max:
                        s.append(str(np.rint(f1)) + str(np.rint(f2)) + str(np.rint(t_delta))) 
        for data in s:
            self.fingerprint.update(data.encode('utf8'))
        if plot:
            print('{} audio-fingerprint: '.format(self.songname))
            print(self.fingerprint.digest())
    
    # Save the AudioFP object to file for later use
    def save_fingerprint(self):
        filename = os.getcwd() + '/audio-fingerprinting/songs/' + self.songname + '.pkl'
        obj_dict = {'songname': self.songname, 'fingerprint': self.fingerprint, 'framerate': self.framerate}
        print('Saving the fingerprint for:', self.songname)
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj_dict, output, pickle.HIGHEST_PROTOCOL)
            
            
# Compare fingerprints of two songs 
def compare_fingerprints(s1, s2):
    jac_sim = calc_jaccard(s1, s2)
    if jac_sim >= 0.9:
        print('{} and {} are identical!'.format(s1.songname, s2.songname))
        print('Jaccard similarity = ', jac_sim)
    elif jac_sim >= 0.1 and jac_sim < 0.9:
        print('{} and {} are quite similar'.format(s1.songname, s2.songname))
        print('Jaccard similarity = ', jac_sim)
    elif jac_sim >= 0.05 and jac_sim < 0.1:
        print('{} and {} might have some similarity'.format(s1.songname, s2.songname))
        print('Jaccard similarity = ', jac_sim)
    else:
        print('{} and {} are different'.format(s1.songname, s2.songname))
        print('Jaccard similarity = ', jac_sim)
        
def calc_jaccard(s1, s2):
    s1_size = s1.fingerprint.count()
    s2_size = s2.fingerprint.count()
    union_s = s1.fingerprint.copy()
    union_s.merge(s2.fingerprint)
    union = union_s.count()
    inter = s1_size + s2_size - union
    jac_sim = inter / union
    return jac_sim
        
# Add Gaussian white noise to a signal
def add_noise(signal, framerate):
    time = np.linspace(0, len(signal)/framerate, len(signal))
    signalpower = signal ** 2
    # Set a target channel noise power to something very noisy
    target_noise_db = int(input('Enter the noise level you want to add in dB: '))
    # Convert to linear units
    target_noise_power = 10 ** (target_noise_db / 10)
    # Generate noise samples
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(target_noise_power), len(signal))
    # Noise up the original signal (again) and plot
    noisy_signal = signal + noise
    return noisy_signal
