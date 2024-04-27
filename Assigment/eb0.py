import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import cv2
import matplotlib.pyplot as plt

# Step 1: Data Collection and Preprocessing
def get_data_generators(data_dir, img_size, batch_size):
    """
    Returns training and validation data generators.
    """
    
    # Augmenting the training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3],
        shear_range=0.3,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Using simple preprocessing for validation data
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Loading data from directories
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'training data'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation data'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

# Step 2: Model Creation and Training
def create_and_train_model(train_generator, val_generator, img_size):
    """
    Creates and trains the model using EfficientNetB0 as the base model.
    """
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_size + (3,))

    # Adding custom dense layers and output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    # Freeze the base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # Train the model
    model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=[early_stopping, lr_plateau]
    )

    # Unfreeze some base model layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Recompile the model for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tune the model
    model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[early_stopping, lr_plateau]
    )
    
    return model

# Step 3: Feature Extraction and Classification
def classify_and_extract_features(image_path, model, img_size, train_generator):
    """
    Classifies the given image and extracts features using the trained model and the base model layers.
    """
    # Read, resize, and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Classify the image
    prediction = model.predict(image)
    label_index = np.argmax(prediction)
    
# Ensure the mapping between label indices and class names is correct
    class_labels = list(train_generator.class_indices.keys())  # Get class labels as a list
    class_labels.sort()  # Ensure the labels are in alphabetical order as they would be in `class_indices`
    
    # Check if the label index is valid and get the label
    if label_index in range(len(class_labels)):
        label = class_labels[label_index]
    else:
        print(f"Warning: Predicted label index {label_index} is invalid.")
        label = None
    
    # Extract features using the base model
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-7].output)
    features = feature_extractor.predict(image)
    
    return label, features

# Step 4: Visual Elements Analysis
# Load a pre-trained Faster R-CNN model
def load_faster_rcnn_model(model_save_path):
    """
    Loads the pre-trained Faster R-CNN model.
    """
    # Load the model
    detector = tf.saved_model.load(model_save_path)
    return detector

# Function to detect objects in an image using Faster R-CNN
def detect_objects(image, detector):
    """
    Detects objects in the given image using the Faster R-CNN model.
    """
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a tensor and add a batch dimension
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform object detection
    detections = detector(input_tensor)

    # Extract detection results
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    # Process results
    detected_objects = []
    for box, cls, score in zip(boxes, classes, scores):
        if score >= 0.5:  # Threshold for detection confidence
            detected_objects.append({
                'box': box,
                'class': cls,
                'score': score
            })

    return detected_objects

# Function to visualize detections
def visualize_detections(image, detections):
    """
    Visualizes detections by drawing bounding boxes and labels on the image.
    """
    # Create a copy of the image
    image_copy = image.copy()

    # Define class labels (customize based on your model's labels)
    class_labels = {
        1: 'Logo',  # Update class IDs and labels as per your model's classes
        2: 'Product',
        # Add more class IDs and labels as needed
    }

    # Draw detections on the image
    for detection in detections:
        box = detection['box']
        cls = detection['class']
        score = detection['score']

        # Draw bounding box
        y_min, x_min, y_max, x_max = box
        start_point = (int(x_min * image.shape[1]), int(y_min * image.shape[0]))
        end_point = (int(x_max * image.shape[1]), int(y_max * image.shape[0]))
        cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

        # Add class label and score
        label = f"{class_labels.get(cls, 'Unknown')} {score:.2f}"
        cv2.putText(image_copy, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title("Detected Objects")
    plt.show()

# Function to analyze visual elements in an image
def analyze_visual_elements(image_path, detector):
    """
    Analyzes visual elements in the given image using the Faster R-CNN model.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Detect objects using Faster R-CNN
    detections = detect_objects(image, detector)
    
    # Visualize detections (optional)
    visualize_detections(image, detections)
    
    # Identify logos and product images from detections
    logos = [detection for detection in detections if detection['class'] == 1]  # Class ID 1 for logos
    products = [detection for detection in detections if detection['class'] == 2]  # Class ID 2 for products
    
    # Placeholder for other visual elements (e.g., text overlays, color schemes)
    # You can implement these as needed using other methods such as OCR and color clustering
    
    detected_elements = {
        'logos': logos,
        'text_overlays': 'Implement text overlay detection here',
        'product_images': products,
        'color_schemes': 'Implement color scheme detection here',
    }
    
    return detected_elements

# Step 5: Integration and Performance Evaluation
def analyze_image(image_path, model, train_generator, img_size, detector):
    """
    Analyzes the image using classification, feature extraction, and visual elements analysis.
    """
    # Classify and extract features
    label, features = classify_and_extract_features(image_path, model, img_size, train_generator)
    print(f"Image classified as: {label}")
    
    # Analyze visual elements
    detected_elements = analyze_visual_elements(image_path, detector)
    print("Detected elements:", detected_elements)
    
    return label, features, detected_elements

def evaluate_model(model, val_generator):
    """
    Evaluates the trained model using the validation data and prints evaluation metrics.
    """
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

# Main script
def main():
    # Define paths and parameters
    data_dir = r'Assignment\data_dir'
    img_size = (224, 224)
    batch_size = 36
    model_save_path = r'Assignment\models\faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8\faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8\saved_model'

    # Load data generators
    train_generator, val_generator = get_data_generators(data_dir, img_size, batch_size)

    # Create and train the model
    model = create_and_train_model(train_generator, val_generator, img_size)

    # Load the pre-trained Faster R-CNN model
    detector = load_faster_rcnn_model(model_save_path)

    # Path to the image for analysis
    image_path = r'C:\Users\abhig\Desktop\Zocket\test5.jpg'
    # Analyze the image
    analyze_image(image_path, model, train_generator, img_size, detector)

    # Evaluate the model
    evaluate_model(model, val_generator)

if __name__ == '__main__':
    main()
