import os
import numpy as np
import tensorflow as tf
import keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Step 1: Data Collection and Preprocessing
data_dir = r'Assignment/data_dir'
batch_size = 28
img_size = (224, 224)

# Image data generator with enhanced augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'Training data'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'Validation data'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Check for class imbalance and handle if needed
class_weights = None
class_distribution = train_generator.class_indices
total_samples = train_generator.samples

# Calculate class weights if any class has a non-zero sample count
class_weights = {}
for cls, num_samples in class_distribution.items():
    if num_samples > 0:
        # Calculate the weight for each class
        class_weights[cls] = total_samples / num_samples
    else:
        print(f"Warning: Class {cls} has zero samples and will be ignored for class weight calculation.")

if not class_weights:
    class_weights = None  # If no class weights were calculated, set to None

# Step 2: Model Selection and Training
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
x = Dropout(0.5)(x)

output_layer = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler function
def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

# Training the model with early stopping and learning rate schedule
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(learning_rate_schedule)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler]
)

# Fine-tune the model by unfreezing the base model layers
for layer in base_model.layers:
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning the model with early stopping and learning rate schedule
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler]
)

# Step 3: Feature Extraction and Classification
def classify_and_extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Classify the image
    prediction = model.predict(image)
    label_index = np.argmax(prediction)
     # Check if label_index is a valid key in train_generator.class_indices
    if label_index in train_generator.class_indices:
        # If valid, get the label
        label = train_generator.class_indices[label_index]
    else:
        # If not valid, handle the case (e.g., by setting to None, warning, etc.)
        print(f"Warning: Predicted label index {label_index} is invalid.")
        label = None
    
    # Extract features using the base model
    feature_extractor = Model(inputs=model.input, outputs=base_model.output)
    features = feature_extractor.predict(image)
    
    return label, features

# Step 4: Visual Elements Analysis
def analyze_visual_elements(image_path):
    # Load the image
    image = cv2.imread(image_path)
     # Load pre-trained YOLOv5 model
    yolo_model = YOLO('yolov5s')  # You can use other YOLOv5 versions such as 'yolov5m', 'yolov5l', etc.
    
    # Perform object detection
    results = yolo_model(image)
    
    # Parse detection results to identify logos and product images
    detected_elements = {
        'logos': [],
        'product_images': []
    }
    
    for result in results:
        # Iterate through detected objects
        for obj in result:
            class_name = obj['name']
            confidence = obj['confidence']
            bbox = obj['bbox']
            
            # Check if the detected class is logo or product image (adjust class names as per your model)
            if class_name == 'logo':
                detected_elements['logos'].append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
            elif class_name == 'product_image':  # Adjust this class name as per your model
                detected_elements['product_images'].append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
    
    # Optionally display the detected objects with bounding boxes
    for obj in results:
        # Draw bounding boxes
        for bbox in obj['bbox']:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Visual Elements")
    plt.show()
    
    return detected_elements

# Step 5: Integration and Performance Evaluation
def analyze_image(image_path):
    label, features = classify_and_extract_features(image_path)
    print(f"Image classified as: {label}")
    
    # Analyze visual elements
    detected_elements = analyze_visual_elements(image_path)
    print("Detected elements:", detected_elements)
    
    return label, features, detected_elements

# Evaluate the model
def evaluate_model():
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Example usage
image_path = '426e417b-a094-46c2-bd39-55a179b42054.png'
analyze_image(image_path)
evaluate_model()
