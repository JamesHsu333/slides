import os
import argparse
import logging
from glob import glob

import numpy as np
import pandas as pd
from imageio.v2 import imread
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.special import gamma
from scipy.optimize import fmin

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr, pearsonr, kendalltau
import umap
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from tqdm import tqdm

# Setup logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def compute_mscn(image, sigma=1.0, C=1.0):
    mu = gaussian_filter(image, sigma=sigma)
    sigma_map = np.sqrt(gaussian_filter((image - mu) ** 2, sigma=sigma))
    mscn = (image - mu) / (sigma_map + C)
    return mscn, mu, sigma_map


def estimate_ggd_params(data):
    data = data.flatten()
    data = data[np.abs(data) > 1e-6]

    def estimate_beta(beta):
        gamma1 = gamma(1 / beta)
        gamma2 = gamma(3 / beta)
        return (gamma2 / gamma1) - (np.mean(np.abs(data)) ** 2 / np.mean(data ** 2))

    best_beta = fmin(lambda b: np.abs(estimate_beta(b[0])), [2], disp=False)[0]
    alpha = np.sqrt(np.mean(data ** 2))
    return alpha, best_beta


def estimate_aggd_params(data):
    data = data.flatten()
    data = data[np.abs(data) > 1e-6]

    left_data = data[data < 0]
    right_data = data[data > 0]

    if len(left_data) == 0 or len(right_data) == 0:
        return np.nan, np.nan, np.nan

    left_std = np.sqrt(np.mean(left_data ** 2))
    right_std = np.sqrt(np.mean(right_data ** 2))

    gamma_hat = left_std / right_std
    r_hat = (np.mean(np.abs(data))) ** 2 / np.mean(data ** 2)
    R_hat = r_hat * ((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)

    def func_to_minimize(beta):
        return (gamma(2 / beta) ** 2) / (gamma(1 / beta) * gamma(3 / beta)) - R_hat

    best_beta = fmin(lambda b: np.abs(func_to_minimize(b[0])), [2], disp=False)[0]
    return left_std, right_std, best_beta


def extract_pairwise_products(mscn):
    h, w = mscn.shape
    return {
        "Horizontal": mscn[:, :-1] * mscn[:, 1:],
        "Vertical": mscn[:-1, :] * mscn[1:, :],
        "Main Diagonal (↘)": mscn[:-1, :-1] * mscn[1:, 1:],
        "Secondary Diagonal (↙)": mscn[:-1, 1:] * mscn[1:, :-1],
    }


def extract_brisque_features(image):
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = rgb2gray(image)
    features = []
    for scale in range(2):
        mscn, _, _ = compute_mscn(image)
        alpha, beta = estimate_ggd_params(mscn)
        features.extend([alpha, beta])

        pair_products = extract_pairwise_products(mscn)
        for product in pair_products.values():
            l_std, r_std, shape = estimate_aggd_params(product)
            features.extend([l_std, r_std, shape])

        image = image[::2, ::2]  # downsample
    return np.array(features)


def train_brisque_model_from_csv(image_dir, csv_file):
    df = pd.read_csv(csv_file)
    df['image_name'] = df['image_name'].astype(str)

    X = []
    y = []
    failed = 0
    total = len(df)

    print("Extracting features from training images...")
    for _, row in tqdm(df.iterrows(), total=total, desc="Training Images", unit="img"):
        image_path = os.path.join(image_dir, row['image_name'])
        if not os.path.exists(image_path):
            failed += 1
            continue
        try:
            img = imread(image_path).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = rgb2gray(img)
            feats = extract_brisque_features(img)
            if np.isnan(feats).any():
                failed += 1
                continue
            X.append(feats)
            y.append(float(row['score']))
        except:
            failed += 1
            continue

    if len(X) == 0:
        raise ValueError("No valid training data extracted. Please check if images are readable and feature extraction succeeds.")

    print(f"Extracted {len(X)} valid samples, skipped {failed} due to NaN or read errors.")

    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svr = SVR(kernel='rbf')
    svr.fit(X_scaled, y)

    joblib.dump((svr, scaler), 'brisque_model.pkl')
    print("Model saved as brisque_model.pkl")

def predict_brisque_score(image, model_path='brisque_model.pkl'):
    svr, scaler = joblib.load(model_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = rgb2gray(image)
    features = extract_brisque_features(image).reshape(1, -1)
    scaled_features = scaler.transform(features)
    return svr.predict(scaled_features)[0]

def visualize_brisque_patch_scores(image_path, patch_size=64, stride=32):
    from imageio.v2 import imread
    img = imread(image_path).astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)

    h, w = img.shape
    heatmap = np.zeros(((h - patch_size) // stride + 1, (w - patch_size) // stride + 1))

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            feats = extract_brisque_features(patch)
            scaled_feats = StandardScaler().fit_transform([feats])
            svr, scaler = joblib.load('brisque_model.pkl')
            if np.isnan(feats).any():
                continue
            score = svr.predict(scaler.transform([feats]))[0]
            heatmap[i // stride, j // stride] = score

    # Resize heatmap to match original image
    from scipy.ndimage import zoom
    heatmap_resized = zoom(heatmap, (stride, stride), order=1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(heatmap_resized, cmap='hot', alpha=0.6)
    plt.colorbar(label='Relative Distortion Score')
    plt.title('Patch-wise BRISQUE Heatmap')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def evaluate_on_testset(image_dir, csv_file, output_csv='evaluation_results.csv'):
    df = pd.read_csv(csv_file)
    df['image_name'] = df['image_name'].astype(str)

    svr, scaler = joblib.load('brisque_model.pkl')

    y_true = []
    y_pred = []
    image_names = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Evaluating test set'):
        image_path = os.path.join(image_dir, row['image_name'])
        if not os.path.exists(image_path):
            continue
        try:
            img = imread(image_path).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = rgb2gray(img)
            feats = extract_brisque_features(img)
            if np.isnan(feats).any():
                continue
            scaled_feats = scaler.transform([feats])
            score = svr.predict(scaled_feats)[0]

            image_names.append(row['image_name'])
            y_true.append(float(row['score']))
            y_pred.append(score)
        except:
            continue

    result_df = pd.DataFrame({
        'image_name': image_names,
        'mos_score': y_true,
        'predicted_score': y_pred
    })
    result_df.to_csv(output_csv, index=False)

    rmse = root_mean_squared_error(y_true, y_pred)
    krcc, _ = kendalltau(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)

    # Prediction vs GT
    plt.figure(figsize=(6, 6))
    sns.regplot(x=y_true, y=y_pred, line_kws={'color': 'red'})
    plt.xlabel('Ground Truth (MOS)')
    plt.ylabel('Predicted Score')
    plt.title(f'RMSE: {rmse:.3f}, KRCC: {krcc:.3f}\nSRCC: {srcc:.3f}, PLCC: {plcc:.3f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_groundtruth.png')
    plt.show()

    # Error distribution
    errors = np.array(y_pred) - np.array(y_true)
    plt.figure(figsize=(6, 4))
    sns.histplot(errors, bins=20, kde=True, color='purple')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (Predicted - Ground Truth)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_error_distribution.png')
    plt.show()

def analyze(image_dir, csv_file):
    df = pd.read_csv(csv_file)
    df['image_name'] = df['image_name'].astype(str)
    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features for analysis", unit="img"):
        image_path = os.path.join(image_dir, row['image_name'])
        if not os.path.exists(image_path):
            continue
        try:
            img = imread(image_path).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = rgb2gray(img)
            feats = extract_brisque_features(img)
            if np.isnan(feats).any():
                continue
            X.append(feats)
            y.append(row['score'])
        except:
            continue

    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    _tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = _tsne.fit_transform(X_scaled)

    _umap = umap.UMAP(n_components=2, random_state=42)
    X_umap = _umap.fit_transform(X_scaled)

    analysis = [
        {'method': 'PCA', 'data': X_pca},
        {'method': 't-SNE', 'data': X_tsne},
        {'method': 'UMAP', 'data': X_umap},
    ]

    for a in analysis:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(a['data'][:, 0], a['data'][:, 1], c=y, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='MOS Score')
        plt.title(f'{a['method'].upper()} of BRISQUE Features')
        plt.xlabel(f'{a['method'].upper()} Dimension 1')
        plt.ylabel(f'{a['method'].upper()} Dimension 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{a['method']}_brisque_embedding.png')
        plt.show()


def visualize_brisque_components(image_path):
    from imageio.v2 import imread
    img = imread(image_path).astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)

    mscn, _, _ = compute_mscn(img)
    pairwise = extract_pairwise_products(mscn)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 4)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap='gray')
    ax0.set_title("Original Image")
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mscn, cmap='bwr')
    ax1.set_title("MSCN Map")
    ax1.axis('off')

    titles = ["Horizontal", "Vertical", "Main Diagonal (↘)", "Secondary Diagonal (↙)"]
    for i in range(4):
        ax = fig.add_subplot(gs[1, i % 4])
        ax.imshow(pairwise[titles[i]], cmap='bwr')
        ax.set_title(f"{titles[i]} Pairwise Product")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BRISQUE Trainer and Evaluator")
    parser.add_argument('--mode', choices=['train', 'predict', 'analyze', 'visualize', 'heatmap', 'evaluate'], required=True, help="Mode")
    parser.add_argument('--csv', type=str, help="CSV file with image_name and score")
    parser.add_argument('--image_dir', type=str, default='images', help="Directory containing images")
    parser.add_argument('--test_image', type=str, help="Single image path for prediction or visualization")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.csv:
            print("Please provide --csv for training.")
        else:
            train_brisque_model_from_csv(args.image_dir, args.csv)

    elif args.mode == 'heatmap':
        if not args.test_image:
            print("Please provide --test_image for patch-wise heatmap.")
        else:
            visualize_brisque_patch_scores(args.test_image)

    elif args.mode == 'evaluate':
        if not args.csv:
            print("Please provide --csv for evaluation.")
        else:
            evaluate_on_testset(args.image_dir, args.csv)
    
    elif args.mode == 'analyze':
        if not args.csv:
            print("Please provide --csv for analysis.")
        else:
            analyze(args.image_dir, args.csv)

    elif args.mode == 'visualize':
        if not args.test_image:
            print("Please provide --test_image for visualization.")
        else:
            visualize_brisque_components(args.test_image)

    elif args.mode == 'predict':
        if not args.test_image or not os.path.exists("brisque_model.pkl"):
            print("Please ensure --test_image is provided and brisque_model.pkl exists.")
        else:
            img = imread(args.test_image).astype(np.float32)
            score = predict_brisque_score(img)
            print(f"Predicted BRISQUE Score: {score:.3f}")