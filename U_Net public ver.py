#%% Imports
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from PIL import Image

#%% DEF

# Root directory
data_path = 'D:/Data/U-NET/split/'

# Image dimensions
image_rows = 512
image_cols = 512

#%% LISTA ARQUIVOS RADIOGRAFIAS

#Cria lista que guarda os nomes dos arquivos com as RADIOGRAFIAS
cxr_flist = []

#Recebe lista contendo os nomes como bytes
cxr = tf.data.Dataset.list_files(str('D:/Data/U-NET/cxr/*.png'), shuffle=False)

#Decodifica os bytes em strings enquanto já adiciona à lista criada inicialmente
for element in cxr.as_numpy_iterator():
    cxr_flist.append(element.decode('UTF-8'))
    print(element.decode('UTF-8'))
    

#%% LISTA ARQUIVOS MÁSCARAS

#Cria lista que guarda os nomes dos arquivos com MÁSCARAS
mask_flist = []

#Recebe lista contendo os nomes como bytes
mask = tf.data.Dataset.list_files(str('D:/Data/U-NET/mask/*mask.png'), shuffle=False)

#Decodifica os bytes em strings enquanto já adiciona à lista criada inicialmente
for element in mask.as_numpy_iterator():
    mask_flist.append(element.decode('UTF-8'))
    print(element.decode('UTF-8'))

#%% LEITURA DAS IMAGENS DAS RADIOGRAFIAS
total = len(cxr_flist)

#Cria listas de Numpy para receber as imagens
cxr_list = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)


#Contador usado para indexar as imagens
i = 0

for file in cxr_flist:
    img = Image.open(file)
    
    #Obtem o menor eixo da imagem (imagem em cinza)
    m = min(img.height,img.width)

    #Corta as bordas
    box = (0,0,m,m)
    img.crop(box)
    
    #Redimensiona para as dimensões padrão usadas na U-Net convencional
    img = img.resize((512,512))
    
    #Converte para NumPy Array
    img_np = np.array(img)
    
    #Se a imagem tiver mais 8 canais, reduza
    if (img_np.ndim==3):
        img_np = img_np[:,:,0]
    
    #Verifica se a imagem foi carregada com os tons invertidos
    canto_NO = img_np[0:512,0:25]
    if (int(canto_NO.mean())>145):
        img_np=255-img_np
    
    print(file+" OK")
    print(type(img))
    
    #Armazena a imagem (já em formato np.array) na lista definitiva
    cxr_list[i] = img_np
    
    #Incrementa o contador
    i = i + 1

#%% LEITURA DAS IMAGENS DAS MÁSCARAS
total = len(mask_flist)

mask_list = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

i = 0

for file in mask_flist:
    img = Image.open(file)
    
    #Obtem o menor eixo da imagem (imagem em cinza)
    m = min(img.height,img.width)

    #Corta as bordas
    box = (0,0,m,m)
    img.crop(box)
    
    #Redimensiona para as dimensões padrão usadas na U-Net convencional
    img = img.resize((512,512))
    
    #Converte para NumPy Array
    img_np = np.array(img)
    
    img_np = img_np.reshape(img_np.shape[0], img_np.shape[1])
    
    print(type(img))
    
    #Armazena a imagem (já em formato np.array) na lista definitiva
    mask_list[i] = img_np
    
    #Incrementa o contador
    i = i + 1

np.save('D:/Data/U-NET/split/experimento/cxr.npy', cxr_list)
np.save('D:/Data/U-NET/split/experimento/mask.npy', mask_list)
print('Saving to .npy files done.')


#%% LER ARQUIVOS NPY

cxr_list = np.load('D:/Data/U-NET/split/experimento/cxr.npy')
mask_list = np.load('D:/Data/U-NET/split/experimento/mask.npy')


#%% RANGE N SHAPE


X = np.asarray(cxr_list, dtype=np.float32)/255
y = np.asarray(mask_list, dtype=np.float32)/255

print(X.shape, y.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

#%% SPLIT DATA

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

#%% INITIALIZE NETWORK

from keras_unet.models import custom_unet

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid'
)


#%%
from keras_unet.utils import get_augmented

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
    data_gen_args = dict(
        zoom_range=0.1
    ))

sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)


#%% 
from tensorflow.keras.callbacks import ModelCheckpoint


model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)

from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded, jaccard_coef
from keras_unet.losses import jaccard_distance

model.compile(
    #optimizer=Adam(), 
    optimizer=SGD(lr=0.001, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[jaccard_coef, iou_thresholded]
)
#%% CARREGA MODELO
from tensorflow.keras.callbacks import ModelCheckpoint

model_filename = 'segm_model_v0.h5'
model.load_weights(model_filename)


#%% RUN!

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=200,
    
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)



from keras_unet.utils import plot_segm_history

plot_segm_history(history,metrics=["iou","jaccard_coef"],)

#%% HISTÓRICO
from keras_unet.utils import plot_segm_history

plot_segm_history(history,metrics=["jaccard_coef"],)


#%% PRED
y_pred = model.predict(x_val[0:5])

yl = y_val[:,:,:,0]

yPR = y_pred[:,:,:,0]


#%% DESENHA DIFERENÇA
#Diferença entre o ground truth e o resultado

mascara = yl.astype(int)
mascara = mascara*10
res = yPR.astype(int)
res = res*5


final = mascara+res

#final=mask^teste
plt.imshow(final[3], cmap = 'plasma')
plt.axis('off')
#skimage.io.imsave("/content/final.bmp",final*255)





