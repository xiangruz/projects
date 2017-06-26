'''

Prelimary code extracted from the code developed for our following paper

Zeng X, Leung M, Zeev-Ben-Mordehai T, Xu M. A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation. 2017. Preprint: arXiv:1706.04970

Authors of the code: Xiangrui Zeng, Min Xu

The current version of this code is mainly for experienced programmers for inspection purpose. It is subject for further updatess to be made more friendly to end users.

Please cite the above paper when this code is used or adapted for your research.

License: GPLv3

'''



'''
#Autoencoder3D

#use DoG for particle picking to extract image patches
#use keras to build a sparse autoencoder (can be multiple layers)
#then cluster the features extracted by the autoencoder

import os
import pickle
import numpy as N
from os.path import join as op_join

data_dir = os.getcwd()

d = pickle.load('data.pickle')

encoder_simple_conv_test(d, out_dir=data_dir)

decode_all_images(d, data_dir)



#d is an image patches data file.
#d is a dictionary consists 'v_siz' and 'vs'.
#d['v_siz'] is an numpy.ndarray specifying the shape of the image patch. For example, d['v_siz'] = array([32,32,32]).
#d['vs'] is a dictionary with keys of uuids specifying each image patch.
#d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.
#d['vs'][an example uuid]['center'] is the center of the image patch in the tomogram. For example, d['vs'][an example uuid]['center'] = [110,407,200].
#d['vs'][an example uuid]['id'] is the specific uuid.
#d['vs'][an example uuid]['v'] are voxel values of the image patch, which is an numpy.ndarray of shape d['v_siz']. 

'''

'''
#EDSS3D


import os
import pickle
import numpy as N

sel_clus = {1:[3,21,28,34,38,39,43,62,63,81,86,88], 2:[15,25,29,33,35,66,79,90,92,98]} #an example of selected clusters for segmentation
#sel_clus is the selected clusters for segmentation, which can be multiple classes.

import os
from os.path import join as op_join

data_dir = os.getcwd()
data_file = op_join(data_dir, 'data.pickle')

# The following files come from the previous Autoencoder3D results
d = pickle.load(data_file)
km = pickle.load(op_join(data_dir, 'clus-center', 'kmeans.pickle'))
cc = pickle.load(op_join(data_dir, 'clus-center', 'ccents.pickle'))
vs_dec = pickle.load(op_join(data_dir, 'decoded', 'decoded.pickle'))

vs_lbl = image_label_prepare(sel_clus, km)
vs_seg = train_label_prepare(vs_lbl=vs_lbl, vs_dec=vs_dec, iso_value=0.5) #iso_value is the mask threshold for segmentation

model_dir = op_join(data_dir, 'model-seg')
if not os.path.isdir(model_dir):    os.makedirs(model_dir)
model_checkpoint_file = op_join(model_dir, 'model-seg--weights--best.h5')
model_file = op_join(model_dir, 'model-seg.h5')

if os.path.isfile(model_file):
    print 'use existing', model_file
    import keras.models as KM
    model = KM.load_model(model_file)
else:
    model = train_validate__reshape(vs_lbl=vs_lbl, vs=d['vs'], vs_seg=vs_seg, model_file=model_file, model_checkpoint_file=model_checkpoint_file)
    model.save(model_file)


#Segmentation prediction on new data

data_dir=os.getcwd() #This should be the new data for prediction
data_file=op_join(data_dir, 'data.pickle')

d = pickle.load(data_file)

prediction_dir = op_join(data_dir, 'prediction')
if not os.path.isdir(prediction_dir):    os.makedirs(prediction_dir)

vs_p = predict__reshape(model, vs={_:d['vs'][_]['v'] for _ in vs_sel})

pickle.dump(vs_p, op_join(prediction_dir,'vs_p.pickle'))

'''

def encoder_simple_conv(img_shape, encoding_dim=32, NUM_CHANNELS=1):

    from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D, Reshape, Flatten
    from keras.models import Sequential, Model
    from keras import regularizers

    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], NUM_CHANNELS)

    input_img = Input(shape=input_shape[1:])
    x = input_img

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)

    encoder_conv_shape = [_.value for _ in  x.get_shape()]
    x = Flatten()(x)

    x = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)         # with sparsity
    
    encoded = x
    encoder = Model(input=input_img, output=encoded)
    print 'encoder', 'input shape', encoder.input_shape, 'output shape', encoder.output_shape

    input_img_decoder = Input(shape=encoder.output_shape[1:])
    x = input_img_decoder
    x = Dense(N.prod(encoder_conv_shape[1:]), activation='relu')(x)
    x = Reshape(encoder_conv_shape[1:])(x)


    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)

    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)

    x = Convolution3D(1, 3, 3, 3, activation='linear', border_mode='same')(x)       # keep the output layer linear activation, so that the image intensity can be negative

    decoded = x
    decoder = Model(input=input_img_decoder, output=decoded)

    autoencoder = Sequential()

    for l in encoder.layers:    autoencoder.add(l)
    for l in decoder.layers:    autoencoder.add(l)

    print('autoencoder layers:')
    for l in autoencoder.layers:    print l.output_shape

    return {'autoencoder':autoencoder, 'encoder':encoder, 'decoder':decoder}



def encoder_simple_conv_test(d, out_dir):
    
    x_keys = [_ for _ in d['vs'] if d['vs'][_]['v'] is not None]
    x_train = [N.expand_dims(d['vs'][_]['v'], -1) for _ in x_keys]
    x_train = N.array(x_train)


    model_dir = op_join(out_dir, 'model')
    if not os.path.isdir(model_dir):    os.makedirs(model_dir)

    model_autoencoder_checkpoint_file = op_join(model_dir, 'model-autoencoder--weights--best.h5')
    model_autoencoder_file = op_join(model_dir, 'model-autoencoder.h5')
    model_encoder_file = op_join(model_dir, 'model-encoder.h5')
    model_decoder_file = op_join(model_dir, 'model-decoder.h5')

    if not os.path.isfile(model_autoencoder_file):
        enc = encoder_simple_conv(img_shape=d['v_siz'])
        autoencoder = enc['autoencoder']

        autoencoder_p = autoencoder

        from keras.optimizers import SGD, Adam
        adam = Adam(lr=0.001, beta_1=0.9)        # choose a proper lr to control convergance speed, and val_loss
        autoencoder_p.compile(optimizer=adam, loss='mean_squared_error')

        if os.path.isfile(model_autoencoder_checkpoint_file):
            print 'loading previous best weights', model_autoencoder_checkpoint_file
            autoencoder_p.load_weights(model_autoencoder_checkpoint_file)

        from keras.callbacks import EarlyStopping, ModelCheckpoint
        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_autoencoder_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        autoencoder_p.fit(x_train, x_train, nb_epoch=500, batch_size=4, shuffle=True, validation_split=0.1, callbacks=[checkpoint, earlyStopping])      # use a large batch size when batch normalization is used

        autoencoder_p.load_weights(model_autoencoder_checkpoint_file)       # we use the best weights for subsequent analysis

        enc['autoencoder'].save(model_autoencoder_file)
        enc['encoder'].save(model_encoder_file)
        enc['decoder'].save(model_decoder_file)
    else:
        import keras.models as KM

        enc = {}
        enc['autoencoder'] = KM.load_model(model_autoencoder_file)
        enc['encoder'] = KM.load_model(model_encoder_file)
        enc['decoder'] = KM.load_model(model_decoder_file)


    x_enc=enc['encoder'].predict(x_train)

    import multiprocessing
    from sklearn.cluster import KMeans
    kmeans_n_init = multiprocessing.cpu_count()

    kmeans = KMeans(n_clusters=100, n_jobs=-1, n_init=100).fit(x_enc)
    x_km_cent = N.array([_.reshape(x_enc[0].shape) for _ in kmeans.cluster_centers_])
    x_km_cent_pred=enc['decoder'].predict(x_km_cent)

    # save cluster info and cluster centers
    clus_center_dir = op_join(out_dir, 'clus-center')
    if not os.path.isdir(clus_center_dir):   os.makedirs(clus_center_dir)

    kmeans_clus = defaultdict(list)
    for i,l in enumerate(kmeans.labels_):   kmeans_clus[l].append(x_keys[i])
    pickle.dump(kmeans_clus, op_join(clus_center_dir, 'kmeans.pickle'))

    ccents = {}
    for i in range(len(x_km_cent_pred)):    ccents[i] = x_km_cent_pred[i].reshape(d['v_siz'])
    pickle.dump(ccents, op_join(clus_center_dir, 'ccents.pickle'))



def decode_all_images(d, data_dir):
    import keras.models as KM
    autoencoder = KM.load_model(op_join(data_dir, 'model', 'model-autoencoder.h5'))
    vs_p = decode_images(autoencoder, d['vs'])

    out_dir = op_join(data_dir, 'decoded')
    if not os.path.isdir(out_dir):   os.makedirs(out_dir)

    pickle.dump(vs_p, op_join(out_dir, 'decoded.pickle'))



def conv_block(x, nb_filter, nb0, nb1, nb2, border_mode='same', subsample=(1, 1, 1), bias=True, batch_norm=False):
    from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D, Reshape, Flatten, Activation
    from keras.layers.normalization import BatchNormalization

    from keras import backend as K
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution3D(nb_filter, nb0, nb1, nb2, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    if batch_norm:
        assert not bias
        x = BatchNormalization(axis=channel_axis)(x)
    else:
        assert bias

    x = Activation('relu')(x)

    return x
    
    
    
def image_label_prepare(sel_clus, km):
    vs_lbl = {}
    for lbl in sel_clus:
        for clus_id in sel_clus[lbl]:
            for k in km[clus_id]:
                vs_lbl[k] = lbl
    return vs_lbl



def train_label_prepare(vs_lbl, vs_dec, iso_value):

    '''
    we only randomly sample a number of image patches in zero class
    this is to speed up calculation and balance class
    '''
    import random
    background_k = [_ for _ in vs_dec if _ not in vs_lbl]
    background_k = random.sample(background_k, len(vs_lbl))

    all_patch_k = set(background_k) | set(vs_lbl.keys())
    vs_seg_t = convert_labels_to_image_channels(vs_lbl=vs_lbl, vs_dec=vs_dec, iso_value=iso_value, all_patch_k=all_patch_k)
    
    n_class = max([vs_lbl[_] for _ in vs_lbl]) + 1
    vs_seg = {}
    for k in vs_seg_t:
        seg_t = vs_seg_t[k]
        s = seg_t[0].shape
        seg = N.zeros((s[0], s[1], s[2], n_class))
        for l in seg_t:       seg[:,:,:,l] = seg_t[l]
        vs_seg[k] = seg

    return vs_seg


def model_simple_upsampling__reshape(img_shape, class_n=None):

    from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D, Reshape, Flatten
    from keras.models import Sequential, Model
    from keras.layers.core import Activation

    NUM_CHANNELS=1
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], NUM_CHANNELS)

    # use relu activation for hidden layer to guarantee non-negative outputs are passed to the max pooling layer. In such case, as long as the output layer is linear activation, the network can still accomodate negative image intendities, just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])
    x = input_img

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)

    x = conv_block(x, 32, 3, 3, 3)

    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)

    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)

    x = Convolution3D(class_n, 1, 1, 1, border_mode='same')(x)
    x = Reshape((N.prod(img_shape), class_n))(x)
    x = Activation('softmax')(x)

    model = Model(input=input_img, output=x)

    print('model layers:')
    for l in model.layers:    print l.output_shape, l.name

    return model




def train_validate__reshape(vs_lbl, vs, vs_seg, model_checkpoint_file, model_file):
    x_keys = vs_seg.keys()
    img_shape = vs[x_keys[0]]['v'].shape

    x = [N.expand_dims(vs[_]['v'], -1) for _ in x_keys]
    x = N.array(x)

    y = [vs_seg[_] for _ in x_keys]
    y = label_reshape(y)
    y = N.array(y)

    class_n = max([vs_lbl[_] for _ in vs_lbl]) + 1

    model = model_simple_upsampling__reshape(img_shape=img_shape, class_n=class_n)

    from keras.optimizers import SGD, Adam
    adam = Adam(lr=0.001, beta_1=0.9)       

    model.compile(optimizer=adam, loss='categorical_crossentropy')

    if os.path.isfile(model_checkpoint_file):
        print 'loading previous best weights', model_checkpoint_file
        model.load_weights(model_checkpoint_file)
  
    class_weight = {}
    for l in range(class_n):    class_weight[l] = float(y.size) / y[:,:,l].sum()
    print 'class_weight', class_weight

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(model_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit(x, y, nb_epoch=10000, batch_size=128, shuffle=True, validation_split=0.1, callbacks=[checkpoint, earlyStopping])

    model.load_weights(model_checkpoint_file)       # we use the best weights for subsequent analysis

    y_p = model.predict(x)
    print 'softmax output check', y_p.shape, y_p.sum(axis=2).min(), y_p.sum(axis=2).max()


    e = model.evaluate(x, y)
    print 'final evaluation', e


    return model



def predict__reshape(model, vs):
    x_keys = [_ for _ in vs]
    x = [N.expand_dims(vs[_], -1) for _ in x_keys]
    x = N.array(x)

    x_shape = x.shape

    y = model.predict(x)
    yr = label_reshape_inverse(y, img_shape=(x_shape[1:4]))
    yr = N.array(yr)

    vs_p = {x_keys[_]:N.squeeze(yr[_,:,:,:,:]) for _ in range(len(x_keys))}
    vs_p = predict__keep_max(vs_p)

    return vs_p


def convert_labels_to_image_channels(vs_lbl, vs_dec, iso_value, all_patch_k=None):
    if all_patch_k is None:     all_patch_k = vs_lbl.keys()
   
    vs_seg = {}
    for k in all_patch_k:
        v = vs_dec[k]
        s = v.shape
        seg = {}
        seg[0] = N.ones(s)              # background channel
        if k in vs_lbl:
            l = vs_lbl[k]
            assert  l > 0       # make sure channel 0 is for background
            seg_v = (-v) > iso_value           # important: watch out signs!!!
            seg[l] = seg_v
            seg[0][seg_v>0.5] = 0           # mask out the corresponding background region
        vs_seg[k] = seg

    return vs_seg



def label_reshape(y):
    sample_n = len(y)
    vr = [None] * sample_n
    for i in range(sample_n):
        vs = y[i]
        vs_s = vs.shape

        cls_n = vs.shape[3]
        vs_f = N.zeros((N.prod(vs_s[0:3]), cls_n))
        for l in range(cls_n):            vs_f[:,l] = vs[:,:,:,l].flatten()
        vr[i] = vs_f

    return vr
    

def label_reshape_inverse(y, img_shape):
    assert y.ndim == 3
    sample_n = len(y)
    cls_n = y.shape[2]

    vr = [None] * sample_n
    for i in range(sample_n):
        vs = y[i]
        vs_f = N.zeros((img_shape[0], img_shape[1], img_shape[2], cls_n))
        for l in range(cls_n):            vs_f[:,:,:,l] = vs[:,l].flatten().reshape(img_shape)
        vr[i] = vs_f

    return vr


def predict__keep_max(vs):
    vs_m = {}
    for k in vs:
        v = vs[k]
        cls_n = v.shape[3]
        max_l = N.argmax(v, axis=3)
        vm = N.zeros_like(v)
        for l in range(cls_n):
            v_l = N.squeeze(v[:,:,:,l])
            v_l[max_l != l] = 0
            vm[:,:,:,l] = v_l
        vs_m[k] = vm
    return vs_m

