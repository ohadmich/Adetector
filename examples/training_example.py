import matplotlib.pyplot as plt

import adetector as adt

pos_files, music_files, podcast_files = adt.train.list_data()
train_generator, test_generator = adt.train.create_data_generators(pos_files,
                                                                   music_files,
                                                                   data_minutes=20,
                                                                   train_fraction=0.5)
history = adt.train.train_CNN_model(train_generator, epochs=5)

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

loss, acc = adt.train.evaluate_model('model1.hdf5', test_generator)
print('The accuracy on the test set is: ' + str(acc))
plt.show()