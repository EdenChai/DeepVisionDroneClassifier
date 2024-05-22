import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
	'test_images',
	target_size=(50, 50),
	batch_size=1,
	class_mode='binary',
	# shuffle=False
)

# Evaluate model
correct_predictions = 0

for i in range(len(test_generator)):
	# Load a single batch of images and labels
	test_images, test_labels = next(test_generator)

	# Set the tensor for the input data
	interpreter.set_tensor(input_details[0]['index'], test_images)
	interpreter.invoke()

	# Get the prediction
	prediction = interpreter.get_tensor(output_details[0]['index'])
	predicted_label = 1 if prediction > 0.5 else 0

	# Check if the prediction is correct
	if predicted_label == test_labels[0]:
		correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(test_generator)
print(f'Correct: {correct_predictions}')
print(f'Test accuracy: {accuracy:.2f}')