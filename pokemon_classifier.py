"""
Project Description:
--------------------
This project is a machine learning pipeline for recognizing Pokémon in images using handcrafted features. It consists of two main stages:

1. Binary Classification:
   - Determines whether an image contains any Pokémon (versus no Pokémon).

2. Multi-class Classification:
   - If Pokémon are detected, a second classifier identifies the specific Pokémon species present in the image.

Key Components:
- **Data Loading & Preprocessing:** 
  Loads images from two datasets (one with Pokémon and one without) and standardizes them (resizing, bit-depth normalization).
  
- **Feature Extraction:**
  Extracts features using methods such as Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and color histograms. (Originally, SIFT features were also used, but they have been disabled to reduce training time.)
  
- **Dimensionality Reduction (Optional):**
  Uses PCA to reduce the feature vector size for faster training and inference.
  
- **Segmentation:**
  Applies a recursive tiling strategy to split images into candidate regions that might contain a Pokémon.
  
- **Model Training & Hyperparameter Tuning:**
  Trains a binary classifier (to detect Pokémon presence) and several multi-class classifiers (to classify the Pokémon species) using GridSearchCV for hyperparameter tuning.
  
- **Inference Pipeline:**
  Processes new images by segmenting them, extracting features, and then using the trained models to predict which Pokémon (if any) are present.

This code is designed to be modular and easily understandable, making it suitable for experimentation and further development.
"""

import os
import gc
os.environ["OMP_NUM_THREADS"] = '4'
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ======================
# Configuration
# ======================
TARGET_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = 64  # Increased from 32
SIFT_CLUSTERS = 50
SEGMENT_MIN_SIZE = 128  # Minimum size for recursive segmentation
MAX_RECURSION_DEPTH = 4  # Maximum recursion depth

# Toggle PCA usage. (Sometimes disabling PCA helps when using handcrafted features.)
USE_PCA = False  # Try setting to False if needed
# Toggle SIFT
USE_SIFT = True
# Adjust the threshold probability for multi-class predictions during inference.
THRESHOLD_PROB = 0.00  # Lowering from 0.2 may yield more detections

# Toggle hyperparameter tuning
DO_HYPERPARAMETER_TUNING = True

# Path configurations
POKEMON_DIR = 'pokemon-data'
NO_POKEMON_DIR = 'no-pokemon'
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ======================
# Feature Extraction
# ======================
class FeatureExtractor:
    def __init__(self, sift_kmeans=None):
        if USE_SIFT:
            self.sift = cv2.SIFT_create()
        self.kmeans = sift_kmeans
        self.scaler_bin = StandardScaler()
        self.scaler_multi = StandardScaler()

    def extract_basic_features(self, img):
        """Enhanced color-aware feature extraction"""
        # --- HOG Features (Multi-channel) ---
        hog_features = []
        for channel in range(3):
            # Using block normalization 'L2-Hys'
            hog_feat = hog(
                img[:, :, channel],
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm='L2-Hys'
            )
            hog_features.append(hog_feat)
        hog_features = np.concatenate(hog_features)
        
        # --- LBP Features ---
        # Convert to LAB for robust texture representation.
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lbp_features = []
        P = 8  # Number of circularly symmetric neighbor set points
        R = 1  # Radius
        n_bins_lbp = P + 2  # For uniform patterns
        for channel in range(3):
            lbp = local_binary_pattern(lab[:, :, channel], P=P, R=R, method='uniform')
            # Use the appropriate number of bins for uniform LBP
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins_lbp, range=(0, n_bins_lbp))
            # Normalize histogram to have unit sum
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-7)
            lbp_features.append(hist)
        lbp_features = np.concatenate(lbp_features)
        
        # --- Color Histograms ---
        # Compute histograms in both HSV and LAB color spaces.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_features = []
        
        # HSV histograms:
        # Note: OpenCV represents H in the range [0,180] for 8-bit images.
        h_hist, _ = np.histogram(hsv[:, :, 0].ravel(), bins=COLOR_BINS, range=(0, 180))
        s_hist, _ = np.histogram(hsv[:, :, 1].ravel(), bins=COLOR_BINS, range=(0, 256))
        v_hist, _ = np.histogram(hsv[:, :, 2].ravel(), bins=COLOR_BINS, range=(0, 256))
        # Normalize each histogram
        h_hist = h_hist.astype('float') / (h_hist.sum() + 1e-7)
        s_hist = s_hist.astype('float') / (s_hist.sum() + 1e-7)
        v_hist = v_hist.astype('float') / (v_hist.sum() + 1e-7)
        color_features.extend([h_hist, s_hist, v_hist])
        
        # LAB histograms:
        for channel in range(3):
            lab_hist, _ = np.histogram(lab[:, :, channel].ravel(), bins=COLOR_BINS, range=(0, 256))
            lab_hist = lab_hist.astype('float') / (lab_hist.sum() + 1e-7)
            color_features.append(lab_hist)
        color_features = np.concatenate(color_features)
        
        # --- Final Feature Vector ---
        return np.concatenate([hog_features, lbp_features, color_features])

    def extract_sift_features(self, img):
        """Color-enhanced SIFT using dominant color channel.
           Note: Consider using a dense sampling strategy if keypoint detection is too sparse.
        """
        variances = [np.var(img[:, :, i]) for i in range(3)]
        dominant_channel = np.argmax(variances)
        _, des = self.sift.detectAndCompute(img[:, :, dominant_channel], None)
        if des is None or len(des) < 5:
            return np.zeros(SIFT_CLUSTERS)
        clusters = self.kmeans.predict(des)
        return np.bincount(clusters, minlength=SIFT_CLUSTERS)

    def extract_all_features(self, img):
        basic = self.extract_basic_features(img)
        if USE_SIFT:
            sift = self.extract_sift_features(img)
            return np.hstack([basic, sift])
        return np.hstack([basic])


# ======================
# Data Loading
# ======================
def load_images(path, target_size=TARGET_SIZE):
    images = []
    for fname in os.listdir(path):
        img_path = os.path.join(path, fname)
        # Read image with unchanged flag to capture any alpha channels or unusual bit depths
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # If image has 4 channels (e.g., BGRA), convert to BGR
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # Ensure image is in 8-bit format
            if img.dtype != np.uint8:
                img = cv2.convertScaleAbs(img)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
    return images

def load_all_data():
    # Binary data: positive (pokemon) vs. negative (no pokemon)
    pokemon_images = []
    for pname in os.listdir(POKEMON_DIR):
        pokemon_images += load_images(os.path.join(POKEMON_DIR, pname))
    no_pokemon_images = load_images(NO_POKEMON_DIR)
    # Multi-class data: labels for individual pokemon
    multi_images = []
    multi_labels = []
    for pname in os.listdir(POKEMON_DIR):
        imgs = load_images(os.path.join(POKEMON_DIR, pname))
        multi_images += imgs
        multi_labels += [pname] * len(imgs)
    
    return pokemon_images, no_pokemon_images, multi_images, multi_labels

# ======================
# Modified Segmentation
# ======================
def recursive_tiling(image, depth=0, min_size=SEGMENT_MIN_SIZE, current_tiles=None):
    """Recursively splits image into quadrants while keeping all segments"""
    if current_tiles is None:
        current_tiles = []
    
    current_tiles.append(image)
    
    if depth >= MAX_RECURSION_DEPTH or image.shape[0] <= min_size or image.shape[1] <= min_size:
        return current_tiles
    
    half_h = image.shape[0] // 2
    half_w = image.shape[1] // 2
    
    #recursive_tiling(image[:half_h, :half_w], depth+1, min_size, current_tiles)
    #recursive_tiling(image[:half_h, half_w:], depth+1, min_size, current_tiles)
    #recursive_tiling(image[half_h:, :half_w], depth+1, min_size, current_tiles)
    #recursive_tiling(image[half_h:, half_w:], depth+1, min_size, current_tiles)
    
    return current_tiles

def filter_positive_tiles(tiles, classifier, feature_extractor, scaler, pca=None):
    """Filter tiles using binary classifier; apply PCA if provided"""
    positive_tiles = []
    for tile in tiles:
        resized = cv2.resize(tile, TARGET_SIZE) if tile.shape[:2] != TARGET_SIZE else tile
        features = feature_extractor.extract_all_features(resized)
        features = scaler.transform([features])
        if pca is not None:
            features = pca.transform(features)
        if classifier.predict(features)[0] == 1:
            positive_tiles.append(tile)
    return positive_tiles

# ======================
# Training Pipeline
# ======================
def train_sift_codebook(images):
    sift = cv2.SIFT_create()
    descriptors = []
    
    for img in images:
        variances = [np.var(img[:, :, i]) for i in range(3)]
        dominant_channel = np.argmax(variances)
        _, des = sift.detectAndCompute(img[:, :, dominant_channel], None)
        if des is not None:
            descriptors.append(des)
    
    if len(descriptors) == 0:
        raise ValueError("No SIFT descriptors found in any image!")
    
    all_descriptors = np.vstack(descriptors)
    kmeans = MiniBatchKMeans(n_clusters=SIFT_CLUSTERS, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

def main():
    # Load data
    pokemon_images, no_pokemon_images, multi_images, multi_labels = load_all_data()
    
    # Train SIFT codebook
    if USE_SIFT:
        print("Training SIFT codebook...")
        sift_kmeans = train_sift_codebook(pokemon_images + no_pokemon_images)
        joblib.dump(sift_kmeans, os.path.join(MODEL_SAVE_DIR, 'sift_kmeans.pkl'))
    
        # Initialize feature extractor
        fe = FeatureExtractor(sift_kmeans)
    else:
        fe = FeatureExtractor()
    """
    # ======================
    # Binary Classification
    # ======================
    print("\nTraining binary classifier...")
    X_bin = [fe.extract_all_features(img) for img in pokemon_images + no_pokemon_images]
    y_bin = [1] * len(pokemon_images) + [0] * len(no_pokemon_images)
    
    X_bin = fe.scaler_bin.fit_transform(X_bin)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, stratify=y_bin, random_state=42)
    
    #print(f"Train -> no-pokemon: {y_train_bin.count(0)}, pokemon: {y_train_bin.count(1)}")
    #print(f"Test -> no-pokemon: {y_test_bin.count(0)}, pokemon: {y_test_bin.count(1)}")
    #exit()

    pca_bin = None
    if USE_PCA:
        pca_bin = PCA(n_components=0.95, random_state=42)
        X_train_bin = pca_bin.fit_transform(X_train_bin)
        X_test_bin = pca_bin.transform(X_test_bin)
        joblib.dump(pca_bin, os.path.join(MODEL_SAVE_DIR, 'pca_bin.pkl'))
    
    if DO_HYPERPARAMETER_TUNING:
        print("\nTuning binary classifier...")
        param_grid_bin = {
            'n_estimators': [100, 150, 200],
            'max_depth': [None, 10, 20]
        }
        grid_bin = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid_bin, cv=3, scoring='accuracy', n_jobs=-1, verbose=2
        )
        grid_bin.fit(X_train_bin, y_train_bin)
        print("Best parameters for binary classifier:", grid_bin.best_params_)
        binary_clf = grid_bin.best_estimator_
    else:
        binary_clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        binary_clf.fit(X_train_bin, y_train_bin)
    
    y_pred_bin = binary_clf.predict(X_test_bin)
    print("Binary Classifier Performance:")
    print(classification_report(y_test_bin, y_pred_bin))
    joblib.dump(binary_clf, os.path.join(MODEL_SAVE_DIR, 'binary_clf.pkl'))
    
    # Clean unused data
    del X_bin, y_bin, X_train_bin, X_test_bin, y_train_bin, y_test_bin
    del y_pred_bin
    if DO_HYPERPARAMETER_TUNING:
        del grid_bin
    gc.collect()
    """
    # ======================
    # Multi-class Classification
    # ======================
    print("\nTraining multi-class classifiers...")
    le = LabelEncoder()
    y_multi = le.fit_transform(multi_labels)
    
    print("Extracting features...")
    X_multi = [fe.extract_all_features(img) for img in tqdm(multi_images, desc='Processing images')]
    X_multi = fe.scaler_multi.fit_transform(X_multi)
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=42)

    pca_multi = None
    if USE_PCA:
        pca_multi = PCA(n_components=0.95, random_state=42)
        X_train_multi = pca_multi.fit_transform(X_train_multi)
        X_test_multi = pca_multi.transform(X_test_multi)
        joblib.dump(pca_multi, os.path.join(MODEL_SAVE_DIR, 'pca_multi.pkl'))
    
    # Define multi-class models along with their parameter grids.
    models = {
        #'XGBoost': (XGBClassifier(random_state=42), {
        #    'n_estimators': [100, 200],
        #    'learning_rate': [0.01, 0.1],
        #    'max_depth': [3, 5]
        #}),
        #'SVM-RBF': (SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'), {
        #    'C': [0.1, 1, 10],
        #    'gamma': ['scale']
        #}),
        #'kNN': (KNeighborsClassifier(weights='distance'), {
        #    'n_neighbors': [3, 5, 7],
        #    'weights': ['uniform', 'distance']
        #}),
        'RForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {
        'n_estimators': [200, 300],
        'max_depth': [7, 10, 15],
        'min_samples_split': [10],
        'min_samples_leaf': [5],
        'max_features': [None],
        'bootstrap': [True]
        })
    }
    
    for model_name, (model, param_grid) in models.items():
        if DO_HYPERPARAMETER_TUNING:
            print(f"\nTuning {model_name}...")
            grid = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2
            )
            grid.fit(X_train_multi, y_train_multi)
            print(f"Best parameters for {model_name}:", grid.best_params_)
            best_model = grid.best_estimator_
        else:
            best_model = model.fit(X_train_multi, y_train_multi)
        
        y_pred = best_model.predict(X_test_multi)
        print(f"\n{model_name} Performance:")
        print(classification_report(le.inverse_transform(y_test_multi), 
                                    le.inverse_transform(y_pred)))
        
        joblib.dump(best_model, os.path.join(MODEL_SAVE_DIR, f'multi_clf_{model_name.lower()}.pkl'))
    
    joblib.dump(fe.scaler_bin, os.path.join(MODEL_SAVE_DIR, 'scaler_bin.pkl'))
    joblib.dump(fe.scaler_multi, os.path.join(MODEL_SAVE_DIR, 'scaler_multi.pkl'))
    joblib.dump(le, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
    
    print("\nTraining complete! Models saved to:", MODEL_SAVE_DIR)

# ======================
# Inference Pipeline
# ======================
class PokemonDetector:
    def __init__(self, model_name='xgboost'):
        if USE_SIFT:
            self.sift_kmeans = joblib.load(os.path.join(MODEL_SAVE_DIR, 'sift_kmeans.pkl'))
            self.fe = FeatureExtractor(self.sift_kmeans)
        else:
            self.fe = FeatureExtractor()
        self.binary_clf = joblib.load(os.path.join(MODEL_SAVE_DIR, 'binary_clf.pkl'))
        self.multi_clf = joblib.load(os.path.join(MODEL_SAVE_DIR, f'multi_clf_{model_name.lower()}.pkl'))
        self.le = joblib.load(os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
        self.scaler_bin = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler_bin.pkl'))
        self.scaler_multi = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler_multi.pkl'))
        
        self.use_pca = USE_PCA
        if self.use_pca:
            self.pca_bin = joblib.load(os.path.join(MODEL_SAVE_DIR, 'pca_bin.pkl'))
            self.pca_multi = joblib.load(os.path.join(MODEL_SAVE_DIR, 'pca_multi.pkl'))
        
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_tiles = recursive_tiling(img_rgb)
        
        candidate_tiles = all_tiles
        #filter_positive_tiles(
        #    all_tiles, self.binary_clf, self.fe, self.scaler_bin,
        #    pca=self.pca_bin if self.use_pca else None
        #)
        # Further filter tiles that are too small
        #candidate_tiles = [t for t in candidate_tiles if t.shape[0] >= SEGMENT_MIN_SIZE and t.shape[1] >= SEGMENT_MIN_SIZE]
        predictions = []
        for tile in candidate_tiles:
            tile_resized = cv2.resize(tile, TARGET_SIZE)
            features = self.fe.extract_all_features(tile_resized)
            features = self.scaler_multi.transform([features])
            if self.use_pca:
                features = self.pca_multi.transform(features)
            
            pred_probs = self.multi_clf.predict_proba(features)[0]
            pred = np.argmax(pred_probs)
            
            class_labels = self.le.inverse_transform(np.arange(len(pred_probs)))  # Get class labels
    
            for label, prob in zip(class_labels, pred_probs):
                print(f"Class: {label}, Probability: {prob:.4f}")

            if pred_probs[pred] >= THRESHOLD_PROB:
                prediction = self.le.inverse_transform([pred])[0]
                predictions.append(prediction)
                print(prediction, pred_probs[pred])
                #plt.figure()
                #plt.imshow(tile) 
                #plt.show()
        
        return list(set(predictions))

if __name__ == '__main__':
    # Train and save models (run once)
    main()
    
    # Example usage
    detector = PokemonDetector(model_name='rforest')
    
    #print("Predictions on test_image0.png:", detector.predict('test_image0.png'))
    #print("Predictions on test_image1.png:", detector.predict('test_image1.png'))
    #print("Predictions on test_image2.png:", detector.predict('test_image2.png'))
    #print("Predictions on test_image3.png:", detector.predict('test_image3.png'))
    #print("Predictions on test_image4.png:", detector.predict('test_image4.png'))
    #print("Predictions on test_image5.png:", detector.predict('test_image5.png'))
    
    print("Predictions on test_squirtle.png:", detector.predict('test_squirtle.png'))
    #print("Predictions on test_pikachu.png:", detector.predict('test_pikachu.png'))
    print("Predictions on test_raichu.png:", detector.predict('test_raichu.png'))
    #print("Predictions on test_bulb_pika.png:", detector.predict('test_bulb_pika.png'))
    #print("Predictions on test_wartotle.png:", detector.predict('test_wartotle.png'))
    print("Predictions on test_charmander.png:", detector.predict('test_charmander.png'))