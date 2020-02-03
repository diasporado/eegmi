import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, center_loss=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.center_loss = center_loss

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # (1024, 250, 6, 7, 9) or (1024, 250, 22)
        y = np.empty((self.batch_size), dtype=int)
        random_y= np.random.rand(self.batch_size, 1)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            crop = ID[0]
            trial = ID[1]
            # Store sample
            X[i,] = self.data[crop][trial]
            # Store class
            y[i] = self.labels[trial]

        y_classes = keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.center_loss:
            return [X, y], [y_classes, random_y]
        return X, y_classes