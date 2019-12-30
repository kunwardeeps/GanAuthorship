
# coding: utf-8

# In[9]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

# In[2]:


input_shape = (32, 32, 3)
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))
sgd = SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])


# In[3]:


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[4]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/ufrc/cis6930/kunwardeep.singh/GanGenerator/data1/data/train',  # this is the target directory
        target_size=(32, 32),  
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/ufrc/cis6930/kunwardeep.singh/GanGenerator/data1/data/validation',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')


# In[10]:


csv_logger = CSVLogger('log_cpu.csv', append=True, separator=';')
model.fit_generator(
        train_generator,
        steps_per_epoch=30000 // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=3000 // batch_size,
        callbacks=[csv_logger])
model.save_weights('gans_real_classifier_cpu.h5')  # always save your weights after training or during training

