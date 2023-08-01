

import torch
import wave
class Silero_vad :

    def __init__(self, SAMPLING_RATE = 16000):
        if(SAMPLING_RATE != 16000 and SAMPLING_RATE != 8000):
            print("SAMPLE RATE MUST BE SET TO 16000 or 8000!!!!!!!!!!!!!!!!!")
            
        self.SAMPLING_RATE = SAMPLING_RATE
        self.windowSize = 512 if SAMPLING_RATE == 16000 else 256
        torch.set_num_threads(1)

        self.USE_ONNX = False # change this to True if you want to test onnx model
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad')

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

    def detect(self, wavfile):
        with wave.open( wavfile) as fd:
            params = fd.getparams()
            frames = fd.readframes(10000000)

        wav = self.read_audio(wavfile, sampling_rate=self.SAMPLING_RATE)
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.SAMPLING_RATE)
        speech_timestamps = self.frames2seconds(speech_timestamps)
        return speech_timestamps
    
    def frames2seconds(self,timestamps):
        converted = []
        for timestamp in timestamps:
            start = timestamp['start']/self.SAMPLING_RATE
            end = timestamp['end']/self.SAMPLING_RATE
            converted.append({'start':start, 'end':end})
        return converted
    
    def detectProbs(self, wavfile, thresh = 0.35, silence = 0.2):
        with wave.open( wavfile) as fd:
            params = fd.getparams()
            frames = fd.readframes(10000000)



        wav = self.read_audio(wavfile, sampling_rate=self.SAMPLING_RATE)

        #get list of voice probablities in 512 chunks
        speech_probs = []
        window_size_samples = self.windowSize # use 256 for 8000 Hz model
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i+window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_prob = self.model(chunk, self.SAMPLING_RATE).item()
            speech_probs.append(speech_prob)

        
        #collect chunks into time segments
        speech_timestamps = [] 
        started = 0 
        start = 0
        for i, speech_prob in enumerate(speech_probs):
            time = (512*i)/self.SAMPLING_RATE
            if not started and speech_prob > thresh: #voice activity. start of segment
                started = 1
                start = time
                probs = [speech_prob]
                silenceChunkCounter = 0
            elif(started and speech_prob > thresh): #voice activity. continue segement
                probs.append(speech_prob)
                silenceChunkCounter = 0
            elif(started and speech_prob < thresh): #no voice activity
                silenceChunkCounter += self.windowSize/self.SAMPLING_RATE
                if(silenceChunkCounter>silence): #check if no voice activity for more than silence time
                    end = time
                    avgProb = round(sum(probs)/len(probs),3)
                    maxProb = max(probs)
                    prob = (avgProb + maxProb) / 2 #mean of max and avarge prob as overall probability
                    speech_timestamps.append({"start": start, "end": end, "prob": prob})#, "avgProb": avgProb, "maxProb": maxProb})
                    started = 0
                
        return speech_timestamps
        
if __name__ == "__main__":
    #reads a 16000khz or 8000khz wav file and performs voice activity detection
    #can return list of timestamps, or list of timestamps with probabilities
    #thesholds and silence patience can be adjusted with detectProbs
    vad = Silero_vad()
   
    wavPath = "audio/out000.wav"
    timestamps = vad.detect(wavPath)
    print(timestamps)
    
    speech_timestamps = vad.detectProbs(wavPath)
    print(speech_timestamps)
