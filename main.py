import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return x_train, x_test, y_train, y_test
    
    def build_model(self, hidden_layers=[128, 64], activation='relu', dropout_rate=0.2):
        model = keras.Sequential()
        
        model.add(layers.Dense(hidden_layers[0], activation=activation, input_shape=(784,)))
        model.add(layers.Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(10, activation='softmax'))
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_model(self, epochs=50, batch_size=128, validation_split=0.1):
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return self.history
    
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, x_data):
        predictions = self.model.predict(x_data)
        return predictions
    
    def predict_classes(self, x_data):
        predictions = self.predict(x_data)
        return np.argmax(predictions, axis=1)
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        y_pred = self.predict_classes(self.x_test)
        y_true = np.argmax(self.y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def classification_report(self):
        y_pred = self.predict_classes(self.x_test)
        y_true = np.argmax(self.y_test, axis=1)
        
        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
        print(report)
    
    def visualize_predictions(self, num_samples=10):
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        samples = self.x_test[indices]
        true_labels = np.argmax(self.y_test[indices], axis=1)
        predictions = self.predict_classes(samples)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = samples[i].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {true_labels[i]}, Pred: {predictions[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

class CNNMNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_preprocess_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return x_train, x_test, y_train, y_test
    
    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return self.history
    
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict_classes(self, x_data):
        predictions = self.model.predict(x_data)
        return np.argmax(predictions, axis=1)

class AdvancedMNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_preprocess_data(self, augment_data=False):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        if augment_data:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
            datagen.fit(x_train)
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return x_train, x_test, y_train, y_test
    
    def build_advanced_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_with_callbacks(self, epochs=50, batch_size=128, validation_split=0.1):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

def run_basic_mnist_classification():
    classifier = MNISTClassifier()
    
    print("Loading and preprocessing MNIST data...")
    classifier.load_and_preprocess_data()
    
    print("Building neural network model...")
    classifier.build_model(hidden_layers=[128, 64], dropout_rate=0.2)
    
    print("Compiling model...")
    classifier.compile_model()
    
    print("Model summary:")
    classifier.model.summary()
    
    print("Training model...")
    classifier.train_model(epochs=50, batch_size=128)
    
    print("Evaluating model...")
    test_loss, test_accuracy = classifier.evaluate_model()
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("Classification Report:")
    classifier.classification_report()
    
    classifier.plot_training_history()
    classifier.plot_confusion_matrix()
    classifier.visualize_predictions()
    
    return classifier

def run_cnn_mnist_classification():
    classifier = CNNMNISTClassifier()
    
    print("Loading and preprocessing MNIST data for CNN...")
    classifier.load_and_preprocess_data()
    
    print("Building CNN model...")
    classifier.build_model()
    
    print("Compiling CNN model...")
    classifier.compile_model()
    
    print("CNN Model summary:")
    classifier.model.summary()
    
    print("Training CNN model...")
    classifier.train_model(epochs=20, batch_size=128)
    
    print("Evaluating CNN model...")
    test_loss, test_accuracy = classifier.evaluate_model()
    print(f"CNN Test Loss: {test_loss:.4f}")
    print(f"CNN Test Accuracy: {test_accuracy:.4f}")
    
    return classifier

def run_advanced_mnist_classification():
    classifier = AdvancedMNISTClassifier()
    
    print("Loading and preprocessing MNIST data with advanced techniques...")
    classifier.load_and_preprocess_data(augment_data=True)
    
    print("Building advanced CNN model...")
    classifier.build_advanced_model()
    
    print("Compiling advanced model...")
    classifier.compile_model(learning_rate=0.001)
    
    print("Advanced Model summary:")
    classifier.model.summary()
    
    print("Training advanced model with callbacks...")
    classifier.train_with_callbacks(epochs=50, batch_size=128)
    
    print("Evaluating advanced model...")
    test_loss, test_accuracy = classifier.evaluate_model()
    print(f"Advanced Test Loss: {test_loss:.4f}")
    print(f"Advanced Test Accuracy: {test_accuracy:.4f}")
    
    return classifier

def compare_models():
    print("Comparing different MNIST classification approaches...")
    
    print("\n" + "="*50)
    print("1. Basic Feedforward Neural Network")
    print("="*50)
    basic_classifier = run_basic_mnist_classification()
    basic_loss, basic_acc = basic_classifier.evaluate_model()
    
    print("\n" + "="*50)
    print("2. Convolutional Neural Network")
    print("="*50)
    cnn_classifier = run_cnn_mnist_classification()
    cnn_loss, cnn_acc = cnn_classifier.evaluate_model()
    
    print("\n" + "="*50)
    print("3. Advanced CNN with Regularization")
    print("="*50)
    advanced_classifier = run_advanced_mnist_classification()
    advanced_loss, advanced_acc = advanced_classifier.evaluate_model()
    
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Basic NN - Loss: {basic_loss:.4f}, Accuracy: {basic_acc:.4f}")
    print(f"CNN - Loss: {cnn_loss:.4f}, Accuracy: {cnn_acc:.4f}")
    print(f"Advanced CNN - Loss: {advanced_loss:.4f}, Accuracy: {advanced_acc:.4f}")

def custom_prediction_demo():
    classifier = MNISTClassifier()
    classifier.load_and_preprocess_data()
    classifier.build_model()
    classifier.compile_model()
    classifier.train_model(epochs=20)
    
    random_indices = np.random.choice(len(classifier.x_test), 5)
    test_samples = classifier.x_test[random_indices]
    true_labels = np.argmax(classifier.y_test[random_indices], axis=1)
    
    predictions = classifier.predict_classes(test_samples)
    probabilities = classifier.predict(test_samples)
    
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"True label: {true_labels[i]}")
        print(f"Predicted label: {predictions[i]}")
        print(f"Confidence: {np.max(probabilities[i]):.4f}")
        print(f"All probabilities: {probabilities[i]}")
        print("-" * 30)

if __name__ == "__main__":
    print("MNIST Handwritten Digit Classification")
    print("Choose an option:")
    print("1. Basic Neural Network")
    print("2. Convolutional Neural Network")
    print("3. Advanced CNN with Regularization")
    print("4. Compare All Models")
    print("5. Custom Prediction Demo")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        run_basic_mnist_classification()
    elif choice == "2":
        run_cnn_mnist_classification()
    elif choice == "3":
        run_advanced_mnist_classification()
    elif choice == "4":
        compare_models()
    elif choice == "5":
        custom_prediction_demo()
    else:
        print("Invalid choice. Running basic neural network by default.")
        run_basic_mnist_classification()
