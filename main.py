import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
from scipy.signal import hann
from scipy.fft import fft


plt.close('all')

#Setup
L = 1024  #Window length
O = L // 2  #Overlap
V = 343  #Sound velocity (m/s)
N = 4  #Number of microphones
D = 0.05  #Microphone distance (m)
SNAP = 25  #Snapshot interval
freq1 = 80  #Frequency range start (Hz)
freq2 = 8000  #Frequency range end (Hz)

#Room setup
room = [4, 6]  #Room dimensions [width, height]
src_pos = [3, 2]  #Source position [x, y]

#Microphone positions
mic_pos = np.zeros((N, 2))
mic_pos[0, :] = [0.2, 3.2]
for i in range(1, N):
    mic_pos[i, :] = [0.2, mic_pos[i-1, 1] - D]

#Plot setup
plt.figure(1, figsize=(8, 6))
plt.hold = True
for i in range(N):
    plt.plot(mic_pos[i, 0], mic_pos[i, 1], 'o', color='black', markersize=7, linewidth=2, label=f'Microphone' if i == 0 else "")
plt.plot(src_pos[0], src_pos[1], '+', color='black', markersize=8, linewidth=2, label='Source')
plt.axis('equal')
plt.xlim([0, room[0]])
plt.ylim([0, room[1]])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Room Setup')
plt.grid(True)
plt.legend()
plt.show()

#Read audio file
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(base_path, "DSB_PHAT_4ch_rn.wav") #<-- with noise
    #sound_path = os.path.join(base_path, "DSB_PHAT_4ch.wav") #<-- without noise
    xsig, fs = sf.read(sound_path)
    print(f"Audio loaded: {xsig.shape} samples at {fs} Hz")
except FileNotFoundError:
    print("Audio file not found")

#Calculate number of time frames
T = int((len(xsig) - L) / O)

#Calculate true DOA angle
DOA = 90 - np.arctan((src_pos[0] - mic_pos[0, 0]) / ((mic_pos[0, 1] + mic_pos[N-1, 1]) / 2 - src_pos[1])) * 180 / np.pi

print(f"True DOA: {DOA:.2f} degrees")

#Initialize variables
a = np.ones((N, 1), dtype=complex)
X = np.zeros((L, N), dtype=complex)
fmin = max(int(np.round(freq1 / (fs / L))), 1)
fmax = int(np.round(freq2 / (fs / L)))
DOA_SCAN = np.arange(-90, 91, 1)  #DOA scanning range
L_DOA_SCAN = len(DOA_SCAN)
SRP_PHATf = np.zeros((fmax - fmin + 1, L_DOA_SCAN))

#Hanning window
wh = np.tile(hann(L).reshape(-1, 1), (1, N))

#Initialize correlation matrix
Rall = np.zeros((N, N, fmax - fmin + 1), dtype=complex)

#Processing variables
k = 0  #Starting index
ss = 1  #Snapshot counter

print(f"Processing {T} time frames...")
print(f"Frequency range: {fmin} to {fmax} bins ({freq1} to {freq2} Hz)")

#Main processing loop
for p in range(T):
    #STFT - apply window and compute FFT
    windowed_signal = xsig[k:k+L, :] * wh
    X = fft(windowed_signal, axis=0)
    
    #Process each frequency bin from fourier transform
    for f in range(fmin, fmax + 1):
        #Update spectral correlation matrix
        Rall[:, :, f - fmin] += np.outer(np.conj(X[f, :]), X[f, :])
        
        #Perform beamforming when snapshot count is reached
        if ss == SNAP:
            R = Rall[:, :, f - fmin]
            
            #Scan all DOA angles
            for is_idx, doa_angle in enumerate(DOA_SCAN):
                # Calculate steering vector
                doa_i = np.sin(doa_angle * np.pi / 180) * fs * D / V
                phase_delays = -1j * 2 * np.pi * doa_i * (f - 1) * np.arange(N) / L
                a = np.exp(phase_delays).reshape(-1, 1)
                
                #SRP-PHAT 
                R_normalized = R / (np.abs(R))   
                SRP_PHATf[f - fmin, is_idx] = np.real(np.conj(a).T @ R_normalized @ a)
    
    #Process results when snapshot is complete
    if ss == SNAP:
        # Sum over all frequencies
        P = np.sum(SRP_PHATf, axis=0)
        
        #Find peak
        im = np.argmax(P)
        DOA_SRP_PHAT = DOA_SCAN[im]
        
        #Plot results
        plt.figure(2)
        plt.clf()
        plt.plot(DOA_SCAN, P, 'b-', linewidth=2)
        plt.axvline(x=DOA_SRP_PHAT, color='r', linestyle='--', linewidth=2, label=f'Estimated DOA: {DOA_SRP_PHAT}°')
        plt.axvline(x=DOA, color='g', linestyle='--', linewidth=2, label=f'True DOA: {DOA:.1f}°')
        plt.xlabel('DOA (degrees)')
        plt.ylabel('SRP-PHAT Power')
        plt.title(f'DOA Estimation - Frame {p+1}')
        plt.grid(True)
        plt.legend()
        plt.xlim([-90, 90])
        
        print(f"Frame {p+1}: Estimated DOA = {DOA_SRP_PHAT}°, True DOA = {DOA:.1f}°, Error = {abs(DOA_SRP_PHAT - DOA):.1f}°")
        
        ss = 1
        Rall = np.zeros((N, N, fmax - fmin + 1), dtype=complex)
        
        
        plt.pause(0.1)
        input("Press Enter to continue...")  
        
    else:
        ss += 1
    
    k += O

print("---------Processing finished-----------")
plt.show()