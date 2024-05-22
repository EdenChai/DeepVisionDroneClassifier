import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
	Step 1 - Model Design
	* CNN architecture: CNN is effective for image classification tasks because it automatically and adaptively learns spatial hierarchies of features.
	* ReLU for hidden layers: It helps the model learn non-linear complexities.
	* Sigmoid for output layers: Since it’s a binary classification, a sigmoid function will output a value between 0 and 1.
	* Optimizer: Adam is a good choice for its adaptive learning rate capabilities.
	* Loss Function: Use binary crossentropy as it is suitable for binary classification tasks.
"""
def createModel():
	# Set the input shape for RGB image (50x50 pixels, 3 color channel)
	input_shape = (50, 50, 3)

	model = keras.Sequential([

		# First convolutional layer: 32 filters of size 3x3, using ReLU activation function
		Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),

		# First Max Pooling layer: a pool size of 2x2
		MaxPooling2D(2, 2),

		# Second convolutional layer: 64 filters of size 3x3, using ReLU activation function
		Conv2D(64, (3, 3), activation='relu'),

		# Second pooling layer: a pool size of 2x2
		MaxPooling2D(2, 2),

		# Flatten the output to feed into a dense layer
		Flatten(),

		# Dense layer: A fully connected layer with 128 units, ReLU activation
		Dense(128, activation='relu'),

		# Dropout layer: for regularization
		Dropout(0.5),

		# Output layer:A single neuron with sigmoid activation to output the probability of the image containing a drone.
		Dense(1, activation='sigmoid')
	])

	# Compile the model with binary crossentropy loss and the Adam optimizer
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

"""
	Step 2 - Data Preparation
	I use `ImageDataGenerator` for easy image loading and augmentation.
"""
def preparationData(train_data_dir, val_data_dir, test_data_dir):
	# Define the image dimensions
	width, height = 50, 50

	# Create data generators for training, validation, and testing
	train_datagen = ImageDataGenerator(rescale=1. / 255)
	val_datagen = ImageDataGenerator(rescale=1. / 255)
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	# Load the data
	train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(height, width), batch_size=32, class_mode='binary')
	val_generator = val_datagen.flow_from_directory(val_data_dir, target_size=(height, width), batch_size=32, class_mode='binary')
	test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(height, width), batch_size=32, class_mode='binary')
	return train_generator, val_generator, test_generator

"""
	Step 3 - Model Training
"""
def trainModel(model, train_generator, val_generator, epochs=30):
	model.fit(train_generator, epochs=epochs, validation_data=val_generator)
	return model

"""
	Step 4 - Model Evaluation
	* Evaluate a model on the test set.
"""
def evaluateModel(model, test_generator):
	# Calculate the number of steps needed to cover all samples
	steps = test_generator.samples / test_generator.batch_size

	# Evaluate the model on the test data
	test_loss, test_acc = model.evaluate(test_generator, steps=steps)
	print(f'Test accuracy: {test_acc:.2f}')

"""
Step 5 - Convert the model to TFLite format
"""
def convertModel(model):
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	return tflite_model

"""
	Step 6 - Save the TFLite model
"""
def saveModel(tflite_model, model_path):
	with open(model_path, 'wb') as f:
		f.write(tflite_model)

########################  EXECUTE FUNCTIONS  ########################

# 1. Create the model
model = createModel()

# Define the paths to the data folders
train_data_dir = 'train_images'
val_data_dir = 'val_images'
test_data_dir = 'test_images'

# 2. Data Preparation
train_generator, val_generator, test_generator = preparationData(train_data_dir, val_data_dir, test_data_dir)

# 3. Model Training
trainModel(model, train_generator, val_generator, 30)

# 4. Evaluate Model
evaluateModel(model, test_generator)

# 5. Convert Model
tflite_model = convertModel(model)

# 6. Save Model
saveModel(tflite_model, 'model.tflite')

