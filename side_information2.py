#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Embedding, BatchNormalization, concatenate
from keras.optimizers import Adam
from keras.layers.merge import dot
from keras.models import Model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


rating_data = pd.read_csv('final_data2.csv')


# In[3]:


rating_data


# In[4]:


rating_data['side_info_user'] = rating_data['energy_consumption_perday']


# In[5]:


from sklearn.utils import shuffle
rating_data = shuffle(rating_data)


# In[6]:


rating_data


# In[7]:


rating_data['side_info_item'] = rating_data['volume(kwh)']


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(rating_data[['user_id','battery_id','side_info_user','side_info_item']], rating_data[['rating']], test_size=0.2, random_state=42)


# In[9]:


from sklearn.model_selection import KFold,cross_val_score
kfold = KFold(n_splits=3,shuffle=False)
for train,test in 


# In[ ]:





# In[14]:


num_users = 150
num_batteries = 39


# In[15]:


aux_input_user = Input(shape=(1,))
aux_dense1 = Dense(128, activation='relu')(aux_input_user)
aux_dense1 = Dropout(0.3)(aux_dense1)
aux_dense_user = Dense(64, activation='relu')(aux_dense1)
aux_dense_user.shape


# In[16]:


aux_input_item = Input(shape=(1,))
aux_dense1 = Dense(128, activation='relu')(aux_input_item)
aux_dense1 = Dropout(0.3)(aux_dense1)
aux_dense_item = Dense(64, activation='relu')(aux_dense1)
aux_dense_item.shape


# In[17]:


latent_factors = 5


# In[18]:


user_input = Input(shape=(1,),name='user_input', dtype='int32')
user_embedding = Embedding(num_users+1, latent_factors, name='user_embedding')(user_input)
user_flat = Flatten(name='user_flat')(user_embedding) #展平也可用 GlobalAveragePooling1D()
user_flat = Dropout(0.3)(user_flat)
user_dense = Dense(256, activation='relu')(user_flat)
user_dense = Dropout(0.3)(user_dense)
user_dense = Dense(128, activation='relu')(user_dense)
user_dense = Dropout(0.3)(user_dense)
user_dense = Dense(64, activation='relu')(user_dense)


# In[19]:


user_dense.shape


# In[20]:


concat_user = concatenate([user_dense,aux_dense_user],axis=-1)


# In[21]:


concat_user.shape


# In[22]:


battery_input = Input(shape=(1,),name='battery_input', dtype='int32')
battery_embedding = Embedding(num_batteries+1, latent_factors, name='battery_embedding')(battery_input)
battery_flat = Flatten(name='battery_flat')(battery_embedding)
battery_flat = Dropout(0.3)(battery_flat)
battery_dense = Dense(256, activation='relu')(battery_flat)
battery_dense = Dropout(0.3)(battery_dense)
battery_dense = Dense(128, activation='relu')(battery_dense)
battery_dense = Dropout(0.3)(user_dense)
battery_dense = Dense(64, activation='relu')(battery_dense)


# In[23]:


concat_battery = concatenate([battery_dense,aux_dense_item],axis=-1)


# In[24]:


product=dot([concat_user, concat_battery], name='product', axes=1)


# In[25]:


model = Model([user_input,aux_input_user,battery_input,aux_input_item],product)


# In[26]:


learning_rate = 0.00005
epochs = 120
batch_size = 256


# In[27]:


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=4,
        verbose=1)
]


# In[28]:


model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])


# In[29]:


history = model.fit([x_train['user_id'], x_train['side_info_user'],x_train['battery_id'],x_train['side_info_item']], y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True, 
                validation_split=0.25)


# In[30]:



model.evaluate([x_test['user_id'],x_test['side_info_user'],x_test['battery_id'],x_test['side_info_item']],y_test)


# In[31]:


import matplotlib.pyplot as plt


# In[32]:


plt.plot(history.epoch,history.history['loss'])
plt.plot(history.epoch,history.history['val_loss'])


# In[43]:


model.evaluate([x_test['user_id'],x_test['side_info_user'],x_test['battery_id'],x_test['side_info_item']],y_test) #loss,rmse,mae


# In[44]:


y_hat = model.predict([x_test['user_id'],x_test['side_info_user'],x_test['battery_id'],x_test['side_info_item']])


# In[45]:


import numpy as np


# In[46]:


y_true = np.array(y_test.rating)


# # simple neural network

# In[47]:


user_input = Input(shape=(1,),name='user_input', dtype='int32')
user_embedding = Embedding(151, latent_factors, name='user_embedding')(user_input)
user_flat = Flatten(name='user_flat')(user_embedding)
user_flat = Dropout(0.3)(user_flat)


# In[48]:


movie_input = Input(shape=(1,),name='movie_input', dtype='int32')
movie_embedding = Embedding(40, latent_factors, name='movie_embedding')(movie_input)
movie_flat = Flatten(name='movie_flat')(movie_embedding)
movie_flat = Dropout(0.3)(movie_flat)


# In[49]:


product = dot([user_flat, movie_flat], name='product', axes=1)


# In[50]:


dense1 = Dense(512, activation='relu')(product)
dense1 = Dropout(0.3)(dense1)
dense2 = Dense(256, activation='relu')(dense1)
dense2 = Dropout(0.3)(dense2)
dense3 = Dense(128, activation='relu')(dense2)
dense3 = Dropout(0.3)(dense3)
dense4 = Dense(64, activation='relu')(dense3)
dense_out = Dense(1, activation='relu')(dense4)


# In[51]:


model1 = Model([user_input, movie_input], dense_out)


# In[52]:


learning_rate = 0.00005
epochs = 120
batch_size = 256


# In[53]:


model1.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])


# In[54]:


fit = model1.fit([x_train['user_id'], x_train['battery_id']], y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True, 
                validation_split=0.25)


# In[55]:


plt.plot(history.epoch,history.history['loss'],label='side_loss')
plt.plot(history.epoch,history.history['val_loss'],label='side_val_loss')
plt.plot(fit.epoch,fit.history['loss'],label='simple_loss')
plt.plot(fit.epoch,fit.history['val_loss'],label='simple_val_loss')
plt.legend()


# In[56]:


model.evaluate([x_test['user_id'],x_test['side_info_user'],x_test['battery_id'],x_test['side_info_item']],y_test)
#loss,rmse,mae


# In[57]:


model1.evaluate([x_train['user_id'], x_train['battery_id']], y_train)


# In[ ]:




