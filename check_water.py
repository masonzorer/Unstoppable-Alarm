# contains functions for checking for the sound of water running
# records with the microphone and runs the audio through ML model
from Model import model
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import pyaudio
import wave

# method to record audio from microphone
def record_water():

    # set parameters for recording
    filename = 'Audio/water.wav'
    channels = 1
    sample_rate = 41000
    chunk_size = 1024
    duration = 5

    # set up audio stream
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    # record audio
    print("Recording...")
    frames = []

    for i in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # convert audio into a wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# method to check if water is running
def check():
    # record audio
    record_water()

    # load audio and convert to spectrogram
    waveform, sample_rate = torchaudio.load('Audio/water.wav', normalize=True)
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=41000, 
        n_fft=2048,
        hop_length=256,
        n_mels=64
    )
    spectrogram = mel_spectrogram_transform(waveform)
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    print(spectrogram.shape)

    # clip spectrogram to the proper size for the model
    if spectrogram.shape[2] > 800:
            spectrogram = spectrogram[:, :, :800]
    else:
        spectrogram = torch.nn.functional.pad(spectrogram, (0, 800 - spectrogram.shape[2]))

    # add a batch dimension
    spectrogram = spectrogram.unsqueeze(0)

    # load model
    CNN_model = model.CNN()
    CNN_model.load_state_dict(torch.load('Model/model-88.46-best.pth'))
    CNN_model.eval()

    # get prediction
    with torch.no_grad():
        outputs = CNN_model(spectrogram)
        _, predicted = torch.max(outputs.data, 1)
    
    # print prediction
    if predicted == 1:
        print("Water is running.")
        return True
    else:
        print("Water is not running.")
        return False

# main for testing
def main():
    check()

if __name__ == "__main__":
    main()

    