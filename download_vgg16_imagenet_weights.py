import os
from tensorflow.keras.applications import VGG16

if __name__ == "__main__":
    vgg = VGG16(weights='imagenet', include_top=False)
    checkpoint_dir = "/opt/algorithm/vgg16"
    vgg.save_weights(checkpoint_dir)
