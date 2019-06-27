import matplotlib.pyplot as plt
import train

pos_files, music_files, podcast_files = train.list_data()
train_generator, test_generator = train.create_data_generators(pos_files,
                                                               music_files,
                                                               data_minutes=20,
                                                               train_fraction=0.5)
history = train.train_CNN_model(train_generator, epochs=5)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history['loss'], linewidth=3, color = 'b')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1,2,2)
plt.plot(history['acc'], linewidth=3, color = 'g')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.tight_layout()
plt.show()