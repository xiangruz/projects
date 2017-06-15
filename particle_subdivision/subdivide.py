'''
Core code extracted from the code used from our following paper

Xu M, Chai X, Muthakana H, Liang X, Yang G, Zeev-Ben-Mordehai T, Xing E. Deep learning based subdivision approach for large scale macromolecules structure recovery from electron cryo tomograms. Preprint: arXiv:1701.08404.  ISMB 2017 (acceptance rate 16%), Bioinformatics doi:10.1093/bioinformatics/btx230 

The current version of this code is mainly for experienced programmers for inspection purpose. It is subject for further updatess to be made more friendly to end users.

Please cite the paper when this code is being used or adapted for your research publication.

License: GPLv3

'''

import numpy as N

from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, merge, ZeroPadding3D, AveragePooling3D, Dropout, Flatten, Activation
import keras.models as KM


def inception3D(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    m = Convolution3D(32, 5, 5, 5, subsample=(1, 1, 1), activation='relu', border_mode='valid', input_shape=())(inputs)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='same')(m)

    # inception module 0
    branch1x1 = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch3x3_reduce = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch3x3 = Convolution3D(64, 3, 3, 3, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch3x3_reduce)
    branch5x5_reduce = Convolution3D(16, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch5x5 = Convolution3D(32, 5, 5, 5, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch5x5_reduce)
    branch_pool = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), border_mode='same')(m)
    branch_pool_proj = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch_pool)
    m = merge([branch1x1, branch3x3, branch5x5, branch_pool_proj], mode='concat', concat_axis=-1)

    m = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), border_mode='valid')(m)
    m = Flatten()(m)
    m = Dropout(0.7)(m)

    # expliciately seperate Dense and Activation layers in order for projecting to structural feature space
    m = Dense(num_labels, activation='linear')(m)
    m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod

def dsrf3D(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)    
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)    
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Flatten(name='flatten')(m)
    m = Dense(512, activation='relu', name='fc1')(m)
    m = Dense(512, activation='relu', name='fc2')(m)
    m = Dense(num_labels, activation='softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod


def compile(model, num_gpus=1):

    if num_gpus > 1:
        import keras_extras.utils.multi_gpu as KUM
        model = KUM.make_parallel(model, num_gpus)

    import keras.optimizers as KOP
    kop = KOP.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=kop, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(model, dj, pdb_id_map, nb_epoch):
    dl = list_to_data(dj, pdb_id_map)

    model.fit(dl['data'], dl['labels'], nb_epoch=nb_epoch, shuffle=True, validation_split=validation_split)


def train_validation(model, dj, pdb_id_map, nb_epoch, validation_split):
    from sklearn.model_selection import train_test_split
    sp = train_test_split(dj, test_size=validation_split)
    train_dl = list_to_data(sp[0], pdb_id_map)
    validation_dl = list_to_data(sp[1], pdb_id_map)

    model.fit(train_dl['data'], train_dl['labels'], validation_data=(validation_dl['data'], validation_dl['labels']), nb_epoch=nb_epoch, shuffle=True)



def predict(model, dj):

    data = list_to_data(dj)
    pred_prob = model.predict(data)      # predicted probabilities
    pred_labels = pred_prob.argmax(axis=-1)

    return pred_labels



def vol_to_image_stack(vs):
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels=1

    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels), dtype=N.float32)

    for i,v in enumerate(vs):        sample_data[i, :, :, :, 0] = v

    return sample_data

def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p:i for i,p in enumerate(pdb_ids)}
    return m

def list_to_data(dj, pdb_id_map=None):
    re = {}

    re['data'] = vol_to_image_stack(vs=[_['v'] for _ in dj])

    if pdb_id_map is not None:
        labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        from keras.utils import np_utils
        labels = np_utils.to_categorical(labels, len(pdb_id_map))
        re['labels'] = labels

    return re



if __name__ == '__main__':
    import pickle
    with open('./data.pickle') as f:     dj = pickle.load(f)

    pdb_id_map = pdb_id_label_map([_['pdb_id'] for _ in dj])

    model = inception3D(image_size=dj[0]['v'].shape[0], num_labels=len(pdb_id_map))

    model = compile(model)
    train_validation(model=model, dj=dj, pdb_id_map=pdb_id_map, nb_epoch=10, validation_split=0.2)



