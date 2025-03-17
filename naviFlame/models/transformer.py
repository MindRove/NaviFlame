import tensorflow as tf
from tensorflow import keras
from keras import layers

CHANNELS = 8
WINDOW_SIZE = 100
CLASSES = 7

def Fsc(U): 
    var_numer = tf.math.reduce_variance(U, axis=-2, keepdims=True)
    numerator = var_numer**(1/2 + 1e-8)
    var_denom = tf.math.reduce_variance(U, axis=-2, keepdims=True)  
    denominator = tf.math.reduce_max(var_denom, axis=None, keepdims=False)+ 1e-8 
    return tf.divide(numerator, denominator)


def SCNet(CHANNELS, WINDOW_SIZE):
    input = keras.Input(shape=(CHANNELS, WINDOW_SIZE, 1))
    U = input  
    Vfsc = Fsc(U)  

    threlu = layers.ThresholdedReLU()(Vfsc)  
    C = threlu.shape[-3] 
    flat = layers.Flatten()(threlu) 

    w2 = layers.Dense(units=2 * C)(flat)  
    relu = layers.Activation('relu')(w2)

    w1 = layers.Dense(units=C)(relu) 
    w1 = tf.reshape(w1, shape=(-1, threlu.shape[-3], threlu.shape[-2], threlu.shape[-1])) 
    V = layers.Activation('softmax')(w1)

    prod = layers.multiply([U, V]) 
    sum = layers.Add()([prod, U])
    U_tilde = sum
    return keras.models.Model(inputs=input, outputs=U_tilde)


def ConvX123(X, size, num_filters):
    leaky = layers.LeakyReLU(alpha=0.1)(X)
    conv = layers.Conv2D(kernel_size=size, strides=1, filters=num_filters, padding='same')(leaky) 
    leaky = layers.LeakyReLU()(conv)
    conv = layers.Conv2D(kernel_size=size, strides=1, filters=X.shape[-1] / 3, padding='same')(leaky)
    dropout = layers.Dropout(0.2)(conv)
    return dropout


def ConvHalve(conv, X):
    concat = layers.Concatenate()([conv, X])
    conv = layers.Conv2D(kernel_size=(1, 1), strides=1, filters=concat.shape[-1] / 2, padding='same')(concat)
    return conv


def SFKNet(input_shape):
    input = keras.Input(shape=(input_shape[1:]))
    X = input
    X1 = ConvX123(X, size=(1, 1), num_filters=X.shape[-1])
    X2 = ConvX123(X, size=(3, 3), num_filters=X.shape[-1])
    X3 = ConvX123(X, size=(5, 5), num_filters=X.shape[-1])
    Z1 = tf.math.reduce_mean(X1, axis=[-3, -2], keepdims=True)
    Z2 = tf.math.reduce_mean(X2, axis=[-3, -2], keepdims=True)
    Z3 = tf.math.reduce_mean(X3, axis=[-3, -2], keepdims=True)

    Z1 = layers.Dense(units=X.shape[-1] / 3)(Z1)
    Z2 = layers.Dense(units=X.shape[-1] / 3)(Z2)
    Z3 = layers.Dense(units=X.shape[-1] / 3)(Z3) 

    U_tilde1 = layers.multiply([X1, Z1])
    U_tilde2 = layers.multiply([X2, Z2])
    U_tilde3 = layers.multiply([X3, Z3]) 
    U_tilde = layers.Concatenate(axis=-1)([U_tilde1, U_tilde2, U_tilde3])
    conv = layers.Conv2D(kernel_size=(1, 1), strides=1, filters=X.shape[-1], padding='same')(U_tilde)

    X_tilde = ConvHalve(conv, X)
    return keras.models.Model(inputs=input, outputs=X_tilde)


def construct_transformer():
    input = keras.Input(shape=(CHANNELS, WINDOW_SIZE, 1)) 

    """
    # augmentation
    #rotate = RotateArmband()(input)
    magn_warp = MyMagnWarping(sigma=0.005, divide=5)(input)
    scal = MyScaling(sigma=0.001)(magn_warp)
    gauss_noise = layers.GaussianNoise(0.005)(scal)
    """
    batch_norm = layers.BatchNormalization()(input)

    # SCNet
    sc = SCNet(CHANNELS, WINDOW_SIZE)(batch_norm)

    # Conv preprocessing
    N = 128
    conv = layers.Conv2D(kernel_size=(3, 3), strides=1, filters=N, padding='same')(sc)
    leaky = layers.LeakyReLU()(conv)

    # SFKNet
    sfk_1 = SFKNet(leaky.shape)(leaky) 
    sfk_2 = SFKNet(sfk_1.shape)(sfk_1)
    sfk = layers.Add()([sfk_1, sfk_2])


    # Dense classification
    global_pool = layers.GlobalAveragePooling2D()(sfk)

    dense = layers.Dense(units=1024)(global_pool)
    dropout = layers.Dropout(0.2)(dense)

    dense = layers.Dense(units=512)(dropout)
    dropout = layers.Dropout(0.2)(dense)

    dense = layers.Dense(units=128)(dropout)
    dropout = layers.Dropout(0.2)(dense)

    dense = layers.Dense(CLASSES)(dropout)
    soft_max = layers.Activation('softmax')(dense)

    return keras.models.Model(inputs=input, outputs=soft_max)
