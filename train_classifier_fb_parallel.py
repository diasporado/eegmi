import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import Model, Sequential, load_model
from keras.layers import Dense,BatchNormalization, Add, \
    Activation,Flatten,Dropout,Reshape,Conv3D,AveragePooling2D, \
    Input, concatenate, LeakyReLU, AveragePooling1D, Embedding, Lambda
from keras import optimizers, callbacks, backend as K

from DepthwiseConv3D import DepthwiseConv3D
from methods import se_block, build_crops
from DataGenerator import DataGenerator
import read_bci_data_fb

'''  Parameters '''
folder_path = 'model_results_fb_parallel_SENet'
use_center_loss = False
use_contrastive_center_loss = False
n_channels = 9
batch_size = 64
all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
n_epoch = 20
early_stopping = 10

'''
Training model for classification of EEG samples into motor imagery classes
'''

def layers(inputs, params=None): 
    pipe1 = DepthwiseConv3D(kernel_size=(1,3,3), strides=(1,1,1), depth_multiplier=64, padding='valid', groups=params['n_channels'])(inputs)
    pipe1 = LeakyReLU(alpha=0.05)(pipe1)
    pipe1 = DepthwiseConv3D(kernel_size=(1,3,3), strides=(1,1,1), depth_multiplier=1, padding='valid', groups=params['n_channels'])(pipe1)
    pipe1 = LeakyReLU(alpha=0.05)(pipe1)
    pipe1 = Conv3D(64, (1,2,3), strides=(1,1,1), padding='valid')(pipe1)
    pipe1 = BatchNormalization()(pipe1)
    pipe1 = LeakyReLU(alpha=0.05)(pipe1)
    pipe1 = Reshape((pipe1.shape[1].value, 64))(pipe1)
    pipe1 = AveragePooling1D(pool_size=(75), strides=(15))(pipe1)

    branch_outputs = []
    pipe2 = Reshape((inputs.shape[1].value, inputs.shape[2].value * inputs.shape[3].value, inputs.shape[4].value))(inputs)
    for i in range(n_channels):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:,:,:,i])(pipe2)
        out = Lambda(lambda x: K.expand_dims(x, -1))(out)
        out = DepthwiseConv2D(kernel_size=(1,42), strides=(1,1), padding='valid', depth_multiplier=128)(out)
        branch_outputs.append(out)
    pipe2 = Add()(branch_outputs)
    # pipe2 = Conv3D(64, (1,6,7), strides=(1,1,1), padding='valid')(inputs)
    pipe2 = BatchNormalization()(pipe2)
    pipe2 = LeakyReLU(alpha=0.05)(pipe2)
    pipe2 = Reshape((pipe2.shape[1].value, 128))(pipe2)
    pipe2 = AveragePooling1D(pool_size=(75), strides=(15))(pipe2)

    pipe = concatenate([pipe1, pipe2], axis=2)
    # pipe = se_block(pipe)
    # pipe = Dense(48)(pipe)
    # pipe = LeakyReLU(alpha=0.05)(pipe)
    pipe = Dropout(0.5)(pipe)
    pipe = Flatten()(pipe)
    return pipe

def l2_loss_func(x):
    epsilon = 1e-8
    output_dim = x[1].shape[-2].value
    centers = x[1][:, 0] # shape: (?, 1024)
    expanded_centers = tf.tile(tf.expand_dims(centers, 1),
                                [1, output_dim, 1])  # (128, 10, 2) repeated centers x128
    expanded_features = tf.tile(tf.expand_dims(x[0], 1),
                                 [1, output_dim, 1])  # (128, 10, 2) repeated features x10
    distance_centers = expanded_features - expanded_centers
    intra_distances = x[0] - centers
    l2_loss_intra = K.sum(K.square(intra_distances), 1, keepdims=True)  # shape: (?, 1)
    inter_distances = K.sum(distance_centers, 1, keepdims=True) - intra_distances
    l2_loss_inter = K.sum(K.square(inter_distances), 1) + epsilon
    return l2_loss_intra / l2_loss_inter

def train(X_list, y, train_indices, val_indices, subject):

    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)

    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': batch_size,
        'n_classes': len(np.unique(y)),
        'n_channels': n_channels,
        'shuffle': True,
        'center_loss': use_center_loss
    }

    training_generator = DataGenerator(X_list, y, train_indices, **params)
    validation_generator = DataGenerator(X_list, y, val_indices, **params)

    steps = len(training_generator)
    output_dim = params['n_classes']
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    opt = optimizers.adam(lr=0.001, beta_2=0.999)
 
    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs, params)
    # pipeline = Dense(64)(pipeline)
    # pipeline = BatchNormalization()(pipeline)
    # ip1 = LeakyReLU(alpha=0.05, name='ip1')(pipeline)
    output = Dense(output_dim, activation=activation)(pipeline)
    
    if use_center_loss or use_contrastive_center_loss:
        lambda_c = 0.25
        input_target = Input(shape=(1,))  # single value ground truth labels as inputs
        centers = Embedding(input_dim=output_dim, output_dim=ip1.shape[-1].value, trainable=True)(input_target)
        if use_center_loss:
            l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([ip1, centers])
        elif use_contrastive_center_loss:
            l2_loss = Lambda(l2_loss_func, name='l2_loss')([ip1, centers])

        model = Model(inputs=[inputs, input_target], outputs=[output, l2_loss])
        model.compile(optimizer=opt,
            loss=[loss, lambda y_true, y_pred: y_pred],
            loss_weights=[1, lambda_c],
            metrics=['accuracy'])
    else:
        model = Model(inputs=inputs, outputs=output)
        opt = optimizers.adam(lr=0.001, beta_2=0.999)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    model.summary()

    cb = [callbacks.ProgbarLogger(count_mode='steps'),
          callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=0.00001),
          callbacks.ModelCheckpoint('./{}/A0{:d}_model.hdf5'.format(folder_path,subject),monitor='loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='val_loss')]

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False, steps_per_epoch=steps,
        workers=4, epochs=n_epoch, verbose=1, callbacks=cb)


def evaluate_model(X_list, y_test, X_indices, subject):

    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)
    trials = X_shape[0]
    crops = len(X_list)
    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': trials,
        'n_classes': len(np.unique(y_test)),
        'n_channels': n_channels,
        'shuffle': False,
        'center_loss': use_center_loss
    }

    actual = [ all_classes[i] for i in y_test ]
    predicted = []
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    output_dim = params['n_classes']
    activation = 'softmax'
    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs, params)
    # pipeline = Dense(64)(pipeline)
    # pipeline = BatchNormalization()(pipeline)
    # ip1 = LeakyReLU(alpha=0.05, name='ip1')(pipeline)
    output = Dense(output_dim, activation=activation)(pipeline)

    if use_center_loss or use_contrastive_center_loss:
        lambda_c = 0.25
        input_target = Input(shape=(1,))  # single value ground truth labels as inputs
        centers = Embedding(input_dim=output_dim, output_dim=ip1.shape[-1].value, trainable=True)(input_target)
        if use_center_loss:
            l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([ip1, centers])
        elif use_contrastive_center_loss:
            l2_loss = Lambda(l2_loss_func, name='l2_loss')([ip1, centers])

        model = Model(inputs=[inputs, input_target], outputs=[output, l2_loss])
        model.load_weights('./{}/{}.hdf5'.format(folder_path,model_name))
    else:
        model = Model(inputs=inputs, outputs=output)
        model.load_weights('./{}/{}.hdf5'.format(folder_path,model_name))
        
    test_generator = DataGenerator(X_list, y_test, X_indices, **params)
    y_pred = model.predict_generator(
        generator=test_generator, verbose=1,
        use_multiprocessing=False, workers=4)
    Y_preds = np.argmax(y_pred, axis=1).reshape(crops, trials)
    Y_preds = np.transpose(Y_preds)

    for j in Y_preds:
        (values,counts) = np.unique(j, return_counts=True)
        ind=np.argmax(counts)
        predicted.append(all_classes[values[ind]])

    kappa_score = metrics.cohen_kappa_score(actual, predicted, labels=all_classes)
    
    confusion_metric =  metrics.confusion_matrix(actual,predicted,labels=all_classes)
    clf_rep = metrics.precision_recall_fscore_support(actual, predicted)
    out_dict = {
         "precision" :clf_rep[0].round(3)
        ,"recall" : clf_rep[1].round(3)
        ,"f1-score" : clf_rep[2].round(3)
        ,"support" : clf_rep[3]
    }
    out_df = pd.DataFrame(out_dict, index = np.sort(all_classes))
    out_df['kappa'] = kappa_score
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 3) if x.name!="support" else  round(x.sum(), 3)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    out_df.to_csv('./{}/{}.csv'.format(folder_path,model_name))
    
    print(metrics.classification_report(actual,predicted))
    print('kappa value: {}'.format(kappa_score))

if __name__ == '__main__': # if this file is been run directly by Python
    
    # load bci competition data set
    
    raw_edf_train, subjects_train = read_bci_data_fb.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    raw_edf_test, subjects_test = read_bci_data_fb.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    # Iterate training and test on each subject separately
    for i in [1,3,4,5,0,2,6,7,8]:
        train_index = subj_train_order[i] 
        test_index = subj_test_order[i]
        np.random.seed(123)
        X, y, _ = read_bci_data_fb.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
        X_list = build_crops(X, increment=5)
        X_indices = []
        crops = len(X_list)
        trials = len(X_list[0])
        for a in range(crops):
            for b in range(trials):
                X_indices.append((a, b))
        X_indices = np.array(X_indices)
        train_indices, val_indices = train_test_split(X_indices, test_size=0.2)

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            train(X_list, y, train_indices, val_indices, i+1)
            del(X)
            del(y)
            del(X_list)
            gc.collect()
            X_test, y_test, _ = read_bci_data_fb.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
            X_list = build_crops(X_test, increment=5)
            X_indices = []
            crops = len(X_list)
            trials = len(X_list[0])
            for a in range(crops):
                for b in range(trials):
                    X_indices.append((a, b))
            evaluate_model(X_list, y_test, X_indices, i+1)
            del(X_test)
            del(y_test)
            del(X_list)
            gc.collect()
