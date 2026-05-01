import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#CARGA DE DATOS Y ETIQUETAS LOCALMENTE

train_ds= tf.keras.preprocessing.image_dataset_from_directory(
  'archive(6)/train', image_size=(48,48), color_mode='grayscale', batch_size=32
)
test_ds= tf.keras.preprocessing.image_dataset_from_directory(
  'archive(6)/test', image_size=(48,48), color_mode='grayscale', batch_size=32
)
val_ds= tf.keras.preprocessing.image_dataset_from_directory(
  'archive(6)/validation', image_size=(48,48), color_mode='grayscale', batch_size=32
)

class_names= train_ds.class_names
print(class_names)
#NORMALIZACIÓN DE LAS IMÁGENES MEDIANTE EXPRESIONES LAMBDA
train_ds= train_ds.map(lambda x,y:(x/255.0,y))
test_ds= test_ds.map(lambda x,y:(x/255.0,y))
val_ds= val_ds.map(lambda x,y:(x/255.0,y))

#RNA
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(48,48,1)),
  # Aumentamos filtros para capturar más detalles
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2), 
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2), 
  tf.keras.layers.Flatten(), 
  # Capa densa más robusta y Dropout para mejorar generalización
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),  
  tf.keras.layers.Dense(len(class_names), activation='softmax')
])
# Usamos una tasa de aprendizaje (learning rate) ligeramente menor para precisión
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
# Entrenamiento
model.fit(train_ds, epochs=8, validation_data=val_ds)
for imgs, labels in test_ds.take(1):
  img= imgs[23]
  real_label= labels[23].numpy()
  img_array= np.expand_dims(img, 0)
  pred= np.argmax(model.predict(img_array), 1)[0]
  plt.imshow(img.numpy(), cmap='gray')
  plt.title(class_names[real_label]+' '+ class_names[pred])
  plt.show()