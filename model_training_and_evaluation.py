import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
train_dir = "/content/drive/MyDrive/Dataset_Training3/Train"
val_dir = "/content/drive/MyDrive/Dataset_Training3/Validation"
test_dir = "/content/drive/MyDrive/Dataset_Training3/Test"
checkpoint_dir = "/content/drive/MyDrive/MobileNetV2_Models"
os.makedirs(checkpoint_dir, exist_ok=True)

# Parameters
img_size = 224
batch_size = 32
epochs = 25
num_classes = 3
initial_epoch = 0

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', shuffle=False)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Custom Callback for Saving Model with Epoch & Val Accuracy
class CustomModelSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('val_accuracy')
        model_name = f"epoch_{epoch+1:02d}_valacc_{acc:.4f}.h5"
        model_path = os.path.join(checkpoint_dir, model_name)
        self.model.save(model_path)
        print(f"✔️ Saved model to {model_path}")

# Resume from latest checkpoint if available
def get_latest_checkpoint():
    models = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not models:
        return None, 0
    models.sort(key=lambda x: int(x.split('_')[1]))
    latest = models[-1]
    epoch_num = int(latest.split('_')[1])
    model.load_weights(os.path.join(checkpoint_dir, latest))
    print(f"🔁 Resuming from checkpoint: {latest}")
    return latest, epoch_num

latest_checkpoint, initial_epoch = get_latest_checkpoint()

# Train
# Train
train_start = time.time()

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[CustomModelSaver()],
    initial_epoch=initial_epoch
)

train_duration = time.time() - train_start
print(f"\n🕒 Total Training Time: {train_duration/60:.2f} minutes ({train_duration:.2f} seconds)")

# Load Best Model (optional - using the last saved one)
best_model_path = os.path.join(checkpoint_dir, sorted(os.listdir(checkpoint_dir))[-1])
model.load_weights(best_model_path)
print(f"✅ Loaded best model from: {best_model_path}")

# Plot Training vs Validation Metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluation
print("\n📊 Evaluating on Test Data")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions and Metrics
pred_probs = model.predict(test_generator)
preds = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n📄 Classification Report:\n")
print(classification_report(y_true, preds, target_names=list(test_generator.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC AUC (One-vs-Rest)
y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
roc_auc = roc_auc_score(y_true_bin, pred_probs, multi_class='ovr')
print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")

# Inference Time
start = time.time()
_ = model.predict(test_generator)
infer_time = time.time() - start
print(f"\n⏱ Inference Time on test set: {infer_time:.2f} seconds")

# Model Size
model.save('mobilenetv2_model_final.h5')
model_size = os.path.getsize('mobilenetv2_model_final.h5') / (1024 * 1024)
print(f"💾 Final Model size: {model_size:.2f} MB")
