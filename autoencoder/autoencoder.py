
'''
use DoG for particle picking to extract small subtomograms
use keras to build a sparse autoencoder (can be multiple layers)
then cluster the features extracted by the autoencoder
reverse cluster centers to get reconstructed noise free images, and pose normalize them
perform clustering of the pose normalized images, to seperate them into several structural classes

Reference:

Zeng X, Leung M, Zeev-Ben-Mordehai T, Xu M. A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation. 2017. Preprint: arXiv:1706.04970

Please cite the above paper when this code is used or adapted for your research.

'''

import os, copy, shutil, uuid, sys
from os.path import join as op_join
from collections import defaultdict
import numpy as N

import autoencoder_util as auto

import os, sys
from os.path import join as op_join

import autoencoder as FEE;      reload(FEE);
import autoencoder_util as auto; reload(auto);


def encoder_simple_conv(img_shape, encoding_dim=32, NUM_CHANNELS=1):
    # workaround for Dropout to work
    #import tensorflow as tf
    #tf.python.control_flow_ops = tf

    from seg_util import conv_block

    from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D, Reshape, Flatten
    from keras.models import Sequential, Model
    from keras import regularizers

    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], NUM_CHANNELS)

    # use relu activation for hidden layer to guarantee non-negative outputs are passed to the max pooling layer. In such case, as long as the output layer is linear activation, the network can still accomodate negative image intendities, just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])
    x = input_img

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), border_mode='same')(x)

    encoder_conv_shape = [_.value for _ in  x.get_shape()]          # x.get_shape() returns a list of tensorflow.python.framework.tensor_shape.Dimension objects
    x = Flatten()(x)

    if False:
        x = Dense(encoding_dim, activation='relu')(x)
    else:
        x = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)         # with sparsity
    
    encoded = x
    encoder = Model(input=input_img, output=encoded)
    print 'encoder', 'input shape', encoder.input_shape, 'output shape', encoder.output_shape

    input_img_decoder = Input(shape=encoder.output_shape[1:])
    x = input_img_decoder
    x = Dense(N.prod(encoder_conv_shape[1:]), activation='relu')(x)
    x = Reshape(encoder_conv_shape[1:])(x)

    if True:
        x = UpSampling3D((2, 2, 2))(x)
        x = conv_block(x, 32, 3, 3, 3)

        x = UpSampling3D((2, 2, 2))(x)
        x = conv_block(x, 32, 3, 3, 3)

        x = Convolution3D(1, 3, 3, 3, activation='linear', border_mode='same')(x)       # keep the output layer linear activation, so that the image intensity can be negative
    else:
        from keras_contrib.layers import Deconvolution3D
        x = Deconvolution3D(1, 1, 1, 1, output_shape=input_shape, subsample=(4,4,4), activation='linear', border_mode='same')(x)         # the correct choice of subsample is important to connect Deconvolution3D with output of last layer

    decoded = x
    decoder = Model(input=input_img_decoder, output=decoded)

    autoencoder = Sequential()
    if True:
        # model with expanded layers, and hopefully allow parallel training
        for l in encoder.layers:    autoencoder.add(l)
        for l in decoder.layers:    autoencoder.add(l)
    else:
        # build encoder according to ~/src/tmp/proj/gan/dcgan.py
        autoencoder.add(encoder)
        autoencoder.add(decoder)

    print('autoencoder layers:')
    for l in autoencoder.layers:    print l.output_shape

    return {'autoencoder':autoencoder, 'encoder':encoder, 'decoder':decoder}


def encoder_simple_conv_test(d, pose, img_org_file, out_dir, clus_num = 50):
    if pose == True:
        tom0 = auto.read_mrc_numpy_vol(img_org_file)
        tom = auto.smooth(tom0, 2.0)
        x_keys = [_ for _ in d['vs'] if d['vs'][_]['v'] is not None]

        x_train_no_pose = [N.expand_dims(d['vs'][_]['v'], -1) for _ in x_keys]
        x_train_no_pose = N.array(x_train_no_pose)
        x_center = [d['vs'][_]['center'] for _ in x_keys]
           

        x_train = []
        default_val = tom.mean()
        x_train_no_pose -= x_train_no_pose.max()
        x_train_no_pose = N.abs(x_train_no_pose)
        
        print 'pose normalizing'
        for i in range(len(x_train_no_pose)):      
            center = x_center[i]
            v = x_train_no_pose[i][:,:,:,0]    
            c = auto.center_mass(v)

            rm = auto.pca(v=v, c=c)['v']             # calculate principal directions
            mid_co = (N.array(v.shape)-1) / 2.0
            loc_r__pn = rm.T.dot(mid_co - c)

            vr = auto.rotate_retrieve(v, tom = tom, rm=rm, center = center, loc_r=loc_r__pn,default_val = default_val)     # pose normalize so that the major axis is along x-axis
            x_train.append(vr)
                      
        x_train = N.array(x_train)
        x_train = N.expand_dims(x_train,axis = 4)

        print 'pose normalization finished'

    else:
        x_keys = [_ for _ in d['vs'] if d['vs'][_]['v'] is not None]

        x_train = [N.expand_dims(d['vs'][_]['v'], -1) for _ in x_keys]
        x_train = N.array(x_train)


    if False:
        # warning, if you normalize here, you need also to normalize when decoding. so it is better not normalize. Use batch normalization in the network instead
        if True:
            x_train -= x_train.mean()
            x_train /= x_train.std()
        else:
            x_train -= x_train.min()
            x_train /= x_train.max()
            x_train -= 0.5
            x_train *= 2


    #print 'x_train.shape', x_train.shape

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
        adam = Adam(lr=0.001, beta_1=0.9, decay = 0.001/500)        # choose a proper lr to control convergance speed, and val_loss
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder_p.compile(optimizer=adam, loss='mean_squared_error')

        if os.path.isfile(model_autoencoder_checkpoint_file):
            print 'loading previous best weights', model_autoencoder_checkpoint_file
            autoencoder_p.load_weights(model_autoencoder_checkpoint_file)
       
        # for debugging
        #import random
        #x_train = N.array(    random.sample([_ for _ in x_train], 1004)   )

        from keras.callbacks import EarlyStopping, ModelCheckpoint
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_autoencoder_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        autoencoder_p.fit(x_train, x_train, nb_epoch=10, batch_size=128, shuffle=True, validation_split=0.1, callbacks=[checkpoint, earlyStopping])      # use a large batch size when batch normalization is used
#        autoencoder_p.fit(x_train, x_train, nb_epoch=500, batch_size=128)

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
    #x_pred=enc['decoder'].predict(x_enc)

    #use kmeans to seperate x_enc into 10 clusters, then decode cluster centers, and patch the decoded cluster centers back to the image, can use mayavi???


    import multiprocessing
    from sklearn.cluster import KMeans
    kmeans_n_init = multiprocessing.cpu_count()

#    import gap
#    import pandas as pd
#    x_enc_df = pd.DataFrame(x_enc)
#    gap_out = gap.gap_statistic(x_enc_df, max_k = clus_num_max)

#    clus_num = range(10, clus_num_max+1)[N.argmax(gap_out.values)]

#    TIF.pickle_dump((x_enc,gap_out,clus_num), op_join(out_dir, 'gap.pickle'))
#    from sklearn import metrics
#    ch_out = [metrics.calinski_harabaz_score(x_enc, KMeans(n_clusters=i, random_state=1).fit(x_enc).labels_) for i in range(10,clus_num_max+1)]
#    clus_num = range(10, clus_num_max + 1)[N.argmax(ch_out)]
#    TIF.pickle_dump((x_enc, ch_out, clus_num), op_join(out_dir, 'ch.pickle'))

    kmeans = KMeans(n_clusters=clus_num, n_jobs=-1, n_init=100).fit(x_enc)
    x_km_cent = N.array([_.reshape(x_enc[0].shape) for _ in kmeans.cluster_centers_])
    x_km_cent_pred=enc['decoder'].predict(x_km_cent)

    # save cluster info and cluster centers
    clus_center_dir = op_join(out_dir, 'clus-center')
    if not os.path.isdir(clus_center_dir):   os.makedirs(clus_center_dir)

    kmeans_clus = defaultdict(list)
    for i,l in enumerate(kmeans.labels_):   kmeans_clus[l].append(x_keys[i])
    auto.pickle_dump(kmeans_clus, op_join(clus_center_dir, 'kmeans.pickle'))

    ccents = {}
    for i in range(len(x_km_cent_pred)):    ccents[i] = x_km_cent_pred[i].reshape(d['v_siz'])
    auto.pickle_dump(ccents, op_join(clus_center_dir, 'ccents.pickle'))
    auto.pickle_dump(x_km_cent, op_join(clus_center_dir, 'ccents_d.pickle'))


def kmeans_centers_plot(clus_center_dir):
    kmeans_clus = auto.pickle_load(op_join(clus_center_dir, 'kmeans.pickle'))
    ccents = auto.pickle_load(op_join(clus_center_dir, 'ccents.pickle'))

    # export slices for visual inspection
    if False:
        for i in ccents:    auto.dsp_cub(ccents[i])
    else:
        clus_center_figure_dir = op_join(clus_center_dir, 'fig')
        if os.path.isdir(clus_center_figure_dir):      shutil.rmtree(clus_center_figure_dir)
        os.makedirs(clus_center_figure_dir)

        # normalize across all images
        min_t = N.min([ccents[_].min() for _ in ccents])
        max_t = N.max([ccents[_].max() for _ in ccents])
        
        assert max_t > min_t
        ccents_t = {_:((ccents[_] - min_t) / (max_t - min_t) ) for _ in ccents}
        ccents_t = ccents

        for i in ccents_t:
            clus_siz = len(kmeans_clus[i])
            t = auto.cub_img(ccents_t[i])['im']
            auto.save_png(t, op_join(clus_center_figure_dir, '%003d--%d.png'%(i,clus_siz)), normalize=False)



if __name__ == "__main__":

    d = auto.pickle_load(sys.argv[1])

    img_org_file = sys.argv[2]

    pose = eval(sys.argv[3])

    clus_num = int(sys.argv[4])
    
    FEE.encoder_simple_conv_test(d = d, pose = pose , img_org_file = img_org_file, out_dir=os.getcwd(), clus_num = 100)

    kmeans_centers_plot(op_join(data_dir, 'clus-center'))




