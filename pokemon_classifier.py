import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ======================
# Configuration
# ======================
TARGET_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = 64 # Was 32
SIFT_CLUSTERS = 50
SEGMENT_MIN_SIZE = 128  # Minimum size for recursive segmentation
MAX_RECURSION_DEPTH = 16  # Maximum recursion depth

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
        self.sift = cv2.SIFT_create()
        self.kmeans = sift_kmeans
        self.scaler_bin = StandardScaler()
        self.scaler_multi = StandardScaler()
        
    def extract_basic_features(self, img):
        # HOG Features
        hog_feat = hog(img, orientations=HOG_ORIENTATIONS,
                      pixels_per_cell=HOG_PIXELS_PER_CELL,
                      cells_per_block=HOG_CELLS_PER_BLOCK,
                      channel_axis=-1)
        
        # LBP Features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=256, range=(0, 256))[0]

        # HSV Color Features
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
        hist_h = np.histogram(hsv[:,:,0], bins=COLOR_BINS, range=(0,180))[0]
        hist_s = np.histogram(hsv[:,:,1], bins=COLOR_BINS, range=(0,256))[0]
        hist_v = np.histogram(hsv[:,:,2], bins=COLOR_BINS, range=(0,256))[0]
        
        return np.concatenate([hog_feat, lbp_hist, hist_h, hist_s, hist_v])
    
    def extract_sift_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Maybe remove
        _, des = self.sift.detectAndCompute(gray, None)
        
        if des is None or len(des) < 5:
            return np.zeros(SIFT_CLUSTERS)
        
        clusters = self.kmeans.predict(des)
        return np.bincount(clusters, minlength=SIFT_CLUSTERS)
    
    def extract_all_features(self, img):
        basic = self.extract_basic_features(img)
        sift = self.extract_sift_features(img)
        return np.hstack([basic, sift])

# ======================
# Data Loading
# ======================
def load_images(path, target_size=TARGET_SIZE):
    images = []
    for fname in os.listdir(path):
        img_path = os.path.join(path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
    return images

def load_all_data():
    # Binary data
    pokemon_images = []
    for pname in os.listdir(POKEMON_DIR):
        pokemon_images += load_images(os.path.join(POKEMON_DIR, pname))
    no_pokemon_images = load_images(NO_POKEMON_DIR)
    
    # Multi-class data
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
    
    # Add current segment to the list
    current_tiles.append(image)
    
    # Base case: stop splitting if below minimum size or max depth
    if depth >= MAX_RECURSION_DEPTH or \
       image.shape[0] <= min_size or \
       image.shape[1] <= min_size:
        return current_tiles
    
    # Split into quadrants
    half_h = image.shape[0] // 2
    half_w = image.shape[1] // 2
    
    # Recursively process all quadrants
    recursive_tiling(image[:half_h, :half_w], depth+1, min_size, current_tiles)
    recursive_tiling(image[:half_h, half_w:], depth+1, min_size, current_tiles)
    recursive_tiling(image[half_h:, :half_w], depth+1, min_size, current_tiles)
    recursive_tiling(image[half_h:, half_w:], depth+1, min_size, current_tiles)
    
    return current_tiles

def filter_positive_tiles(tiles, classifier, feature_extractor, scaler):
    """Filter tiles using binary classifier"""
    positive_tiles = []
    for tile in tiles:
        resized = cv2.resize(tile, TARGET_SIZE)
        features = feature_extractor.extract_all_features(resized)
        features = scaler.transform([features])
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        if des is not None:
            descriptors.append(des)
    
    all_descriptors = np.vstack(descriptors)
    kmeans = MiniBatchKMeans(n_clusters=SIFT_CLUSTERS, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

def main():
    # Load data
    pokemon_images, no_pokemon_images, multi_images, multi_labels = load_all_data()
    
    # Train SIFT codebook
    print("Training SIFT codebook...")
    sift_kmeans = train_sift_codebook(pokemon_images + no_pokemon_images)
    joblib.dump(sift_kmeans, os.path.join(MODEL_SAVE_DIR, 'sift_kmeans.pkl'))
    
    # Initialize feature extractor
    fe = FeatureExtractor(sift_kmeans)
    
    # ======================
    # Binary Classification
    # ======================
    print("\nTraining binary classifier...")
    X_bin = [fe.extract_all_features(img) for img in pokemon_images + no_pokemon_images]
    y_bin = [1]*len(pokemon_images) + [0]*len(no_pokemon_images)
    
    X_bin = fe.scaler_bin.fit_transform(X_bin)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, stratify=y_bin, random_state=42)
    
    binary_clf = RandomForestClassifier(n_estimators=150, random_state=42)
    binary_clf.fit(X_train_bin, y_train_bin)

    y_pred_bin = binary_clf.predict(X_test_bin)
    print("Binary Classifier Performance:")
    print(classification_report(y_test_bin, y_pred_bin))
    joblib.dump(binary_clf, os.path.join(MODEL_SAVE_DIR, 'binary_clf.pkl'))
    
    # ======================
    # Multi-class Classification
    # ======================
    print("\nTraining multi-class classifier...")
    le = LabelEncoder()
    y_multi = le.fit_transform(multi_labels)
    
    X_multi = [fe.extract_all_features(img) for img in multi_images]
    X_multi = fe.scaler_multi.fit_transform(X_multi)
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=42)
    
    multi_clf = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    multi_clf.fit(X_train_multi, y_train_multi)

    # Evaluate Multi-class Classifier
    y_pred_multi_encoded = multi_clf.predict(X_test_multi)
    y_pred_multi = le.inverse_transform(y_pred_multi_encoded)  # Convert back to names

    print("Multi-class Classifier Performance:")
    print(classification_report(le.inverse_transform(y_test_multi), y_pred_multi))

    joblib.dump(multi_clf, os.path.join(MODEL_SAVE_DIR, 'multi_clf.pkl'))
    
    # Save other components
    joblib.dump(fe.scaler_bin, os.path.join(MODEL_SAVE_DIR, 'scaler_bin.pkl'))
    joblib.dump(fe.scaler_multi, os.path.join(MODEL_SAVE_DIR, 'scaler_multi.pkl'))
    joblib.dump(le, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
    
    print("\nTraining complete! Models saved to:", MODEL_SAVE_DIR)

# ======================
# Modified Inference Pipeline
# ======================
class PokemonDetector:
    def __init__(self):
        # Existing initialization
        self.sift_kmeans = joblib.load(os.path.join(MODEL_SAVE_DIR, 'sift_kmeans.pkl'))
        self.fe = FeatureExtractor(self.sift_kmeans)
        self.binary_clf = joblib.load(os.path.join(MODEL_SAVE_DIR, 'binary_clf.pkl'))
        self.multi_clf = joblib.load(os.path.join(MODEL_SAVE_DIR, 'multi_clf.pkl'))
        self.le = joblib.load(os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
        self.scaler_bin = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler_bin.pkl'))
        self.scaler_multi = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler_multi.pkl'))
        
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate all possible tiles recursively
        all_tiles = recursive_tiling(img_rgb)
        
        #for img in all_tiles:
        #    imgplot = plt.imshow(img)
        #    plt.show()

        # Filter tiles using binary classifier
        candidate_tiles = filter_positive_tiles(
            all_tiles, self.binary_clf, self.fe, self.scaler_bin
        )

        # Further filter tiles that are too small
        candidate_tiles = [
            t for t in candidate_tiles
           if t.shape[0] >= SEGMENT_MIN_SIZE and t.shape[1] >= SEGMENT_MIN_SIZE
        ]

        predictions = []
        for tile in candidate_tiles:
            tile_resized = cv2.resize(tile, TARGET_SIZE)
            features = self.fe.extract_all_features(tile_resized)
            features = self.scaler_multi.transform([features])
            
            pred_probs = self.multi_clf.predict_proba(features)[0]
            pred = np.argmax(pred_probs)

            if pred_probs[pred] >= 0.5:
                predictions.append(self.le.inverse_transform([pred])[0])
                #print(predictions[len(predictions)-1])
                #imgplot = plt.imshow(tile)
                #plt.show()
        
        return list(set(predictions))

if __name__ == '__main__':
    # Train and save models (run once)
    #main()
    
    # Example usage
    detector = PokemonDetector()
    print("Predictions:", detector.predict('test_image0.png')) # [Bulbasaur, Pikachu, Charmander, Squirtle]
    print("Predictions:", detector.predict('test_image1.png')) # [Charmander, Squirtle, Bulbasaur]
    print("Predictions:", detector.predict('test_image2.png')) # [Bulbasaur, Ivysaur, Venusaur, Squirtle, Wartotle, Blastoise, Charmander, Charmeleon, Charizard]
    print("Predictions:", detector.predict('test_image3.png')) # [Charmander, Squirtle, Bulbasaur]
    print("Predictions:", detector.predict('test_image4.png')) # [Bulbasaur, Ivysaur, Venusaur, Squirtle, Wartotle, Blastoise, Charmander, Charmeleon, Charizard]