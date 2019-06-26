import adetector as adt

radio_stream_path =  '../Data/sample_audio.wav'
X = adt.core.audio2features(radio_stream_path)
timestamps, probs = adt.core.find_ads(X, T=0.96, n=1, show=True)

print("Ads were detected at the following timestamps:")
print(timestamps)
print("Ad probabilities for these timestamps are:")
print(probs)