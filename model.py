from flask import Flask, request, jsonify, Response, send_file
import zipfile
# from pydub import AudioSegment
# from keyword_spotting_service import Keyword_spotting_Service
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import io
# from werkzeug.utils import secure_filename
import os
import wave
import scipy.io.wavfile
from scipy.io.wavfile import write
# import IPython.display as ipd
import soundfile as sf
import random
import IPython.display as ipd
app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Check if the POST request has the 'audio' file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    sample_rate = 5512
# Load audio signal
    audio_sound, sr = librosa.load(audio_file, sr = sample_rate)
    print(audio_sound.shape)

    FRAME = 512
    HOP = 256

# Return the complex Short Term Fourier Transform
    sound_stft = librosa.stft(audio_sound, n_fft = FRAME, hop_length = HOP)
    # Magnitude Spectrogram
    sound_stft_Magnitude = np.abs(sound_stft)

# Phase spectrogram
    sound_stft_Angle = np.angle(sound_stft)  
    epsilon = 1e-10 # error to introduce
    V = sound_stft_Magnitude + epsilon
    K, N = np.shape(V)
    S = 2
    W, H, cost_function = NMF(V,S,beta = 2, threshold = 0.05, MAXITER = 100)   
    # Return the processed audio as a response

    filtered_spectrograms = []
    for i in range(S):
        # axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}") 
        # Filter eash source components
        WsHs = W[:,[i]]@H[[i],:]
        filtered_spectrogram = W[:,[i]]@H[[i],:] /(W@H) * V 
        # Compute the filtered spectrogram
        D = librosa.amplitude_to_db(filtered_spectrogram, ref = np.max)
        # Show the filtered spectrogram
        # librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet, ax = axs[i])
        
        filtered_spectrograms.append(filtered_spectrogram)    
    reconstructed_sounds = []
    for i in range(S):
        reconstruct = filtered_spectrograms[i] * np.exp(1j*sound_stft_Angle)
        new_sound   = librosa.istft(reconstruct, n_fft = FRAME, hop_length = HOP)
        reconstructed_sounds.append(new_sound)
    for i in range(S):
        print(reconstructed_sounds)
        reconstructed_sounds.append(convert_audio_to_bytes(reconstructed_sounds[i]))

    
    zip_filename = "reconstructed_audio1.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, audio_bytes in enumerate(reconstructed_sounds):
            zipf.writestr(f'reconstructed_audio_{i}.wav', audio_bytes)
    
    
    # Send the zip file as a response
    return send_file(zip_filename, as_attachment=True)

def convert_audio_to_bytes(audio_data):
    # Convert audio data to bytes (e.g., WAV format)
    output_buffer = io.BytesIO()
    # sf.write( output_buffer, value["array"], value["sampling_rate"], format="wav")
    sf.write(output_buffer, audio_data, 5512, format="wav")
    output_buffer.seek(0)
    return output_buffer.read()


# def send_audio_as_response(audio_data):
#     # Convert audio data to the appropriate format (e.g., WAV)
#     # You might need to use a library like librosa to save it as WAV
#     for audio in audio_data:
#         output_bytes1 = convert_audio_to_bytes(audio)
        

#     # output_bytes2 = convert_audio_to_bytes(audio_data[1])

#     # Send the audio data as a response with the appropriate content type
#     return Response(output_bytes1, content_type='audio/wav')


def send_audio_files_as_response(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        audio_responses = []
        for filename in zipf.namelist():
            audio_bytes = zipf.read(filename)
            audio_responses.append(Response(audio_bytes, content_type='audio/wav'))

    return audio_responses














# def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=5512):
#     bytes_wav = bytes()
#     byte_io = io.BytesIO(bytes_wav)
#     write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
#     output_wav = byte_io.read()
#     output, samplerate = sf.read(io.BytesIO(output_wav))
#     return output



def divergence(V,W,H, beta = 2):
    
    """
    beta = 2 : Euclidean cost function
    beta = 1 : Kullback-Leibler cost function
    beta = 0 : Itakura-Saito cost function
    """ 
    
    if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
    
    if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
    
    if beta == 2 : return 1/2*np.linalg.norm(W@H-V)
    
def update_beta_step(V,H,W,beta):
    
    H_new *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
    W_new *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
    return W_new,H_new 

def NMF(V, S, beta = 2,  threshold = 0.05, MAXITER = 5000, display = False ): 
    
    """
    inputs : 
    --------
    
        V         : Mixture signal : |TFST|
        S         : The number of sources to extract
        beta      : Beta divergence considered, default=2 (Euclidean)
        threshold : Stop criterion 
        MAXITER   : The number of maximum iterations, default=1000
        display   : Display plots during optimization : 
        displayEveryNiter : only display last iteration 
                                                            
    
    outputs :
    ---------
      
        W : dictionary matrix [KxS], W>=0
        H : activation matrix [SxN], H>=0
        cost_function : the optimised cost function over iterations
       
   Algorithm : 
   -----------
   
    1) Randomly initialize W and H matrices
    2) Multiplicative update of W and H 
    3) Repeat step (2) until convergence or after MAXITER 
    
       
    """
    counter  = 0
    cost_function = []
    beta_divergence = 1
    beta =2
    
    K, N = np.shape(V)
    
    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))
    
    # # Plotting the first initialization
    # if display == True : plot_NMF_iter(W,H,beta,counter)


    while beta_divergence >= threshold and counter <= MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        # W=update_beta_step(V,H,W,2)
        # H=update_beta_step(V.T,W.T,H.T,2).T
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        print(beta_divergence)
        cost_function.append( beta_divergence )
        
        # if  display == True  and counter%displayEveryNiter == 0  : plot_NMF_iter(W,H,beta,counter)

        counter +=1
    
    if counter -1 == MAXITER : print(f"Stop after {MAXITER} iterations.")
    else : print(f"Convergeance after {counter-1} iterations.")
        
    return W,H, cost_function 



if __name__ == '__main__':
    app.run(debug=True)
