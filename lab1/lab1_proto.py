# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------
import numpy as np
from scipy import signal as sg
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from lab1_tools import trfbank, lifter

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        samples: Array of input samples.
        winlen: Window length in samples.
        winshift: Shift of consecutive windows in samples.
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal.
    """
    num_frames = 1 + int((len(samples) - winlen) / winshift)
    frames = np.zeros((num_frames, winlen))
    
    # Calculate nr of frames that can fit into input samples
    # Iterate over frames, slicing into smaller frames
    for i in range(num_frames):
        start = i * winshift
        end = start + winlen
        frames[i, :] = samples[start:end]
    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = np.array([1, -p]) # Numerator coefficients: define zero location in z-domain, attentuate specific frequency components of input signal.
                          # 1 multiplies x[n], and -p multiplies x[n-1].
    a = [1]               # Denominator coefficient: means that y[n] is not recursively influenced by its past values, which is the case for FIR.

    # lfilter filters data along one-dimension with first order FIR filter.
    output = sg.lfilter(b, a, input, axis=1)
    return output

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    hamming_window = sg.hamming(input.shape[1], sym=0)
    return hamming_window * input

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft_res = fft(input, n=nfft)
    power_spectrum = np.abs(pow(fft_res, 2))
    return power_spectrum

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    mel_filters = trfbank(samplingrate, input.shape[1])
    mel_spectrum = input @ mel_filters.T  # Note .T since mel_filters has shape [N, nfft]
    log_mel_spectrum = np.log(mel_spectrum)
    return log_mel_spectrum

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    cepstral_coeffs = dct(input)[:, :nceps]
    return cepstral_coeffs

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N, M = len(x), len(y)

    # Local distance
    LD = np.zeros((N, M))
    LD = dist(x, y)
    
    # Accumulated distance
    AD = np.zeros((N, M))
    AD[0, 0] = LD[0, 0]
    
    # Fill in the first column and row, respectively
    AD[1:N, 0] = np.cumsum(LD[1:N, 0])
    AD[0, 1:M] = np.cumsum(LD[0, 1:M])
    
    # Fill in rest of matrix based on first row and column
    for i in range(1, N):
        for j in range(1, M):
            AD[i, j] = LD[i, j] + min(AD[i-1, j], AD[i, j-1], AD[i-1, j-1])
    
    # Global dist. is in bottom-right corner of AD 
    d = AD[N-1, M-1] / (N + M)  # Normalize to len(x)+len(y)
    
    # Compute the warping path, choose path with smallest cumulative dist
    path = []
    i, j = N-1, M-1 # Start from bottom-right corner of the AD matrix
    path.append((i, j))
    while i > 0 or j > 0:
        # Edge cases
        if i == 0: 
            j = j - 1 # Can only go left, already at the top
        elif j == 0:
            i = i - 1 # Can only go up, already at left edge
        
        # Inside AD matrix, can go (Up, left), (Left) or (Up).
        else:
            min_value = min(AD[i-1, j], AD[i, j-1], AD[i-1, j-1])
            if min_value == AD[i-1, j-1]: #Up, left 
                i, j = i-1, j-1   
            elif min_value == AD[i, j-1]: #Left
                j = j - 1
            else:                       #Up
                i = i - 1
        path.insert(0, (i, j))
    
    # d only necessary for this exxercise, but LD, AD, path are here as well.
    return d, LD, AD, path