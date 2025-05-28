import os
import glob
import argparse
import numpy as np
import scipy.io
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
import pandas as pd

def test_image(model, image_path):
    img = image.load_img(image_path)
    img = np.array(img)

    if img.shape[0] < 299 or img.shape[1] < 299:
        print(f"⚠️ Image {image_path} is too small. Skipped.")
        return None

    h_patches = img.shape[0] // 299
    w_patches = img.shape[1] // 299
    patches = np.zeros((h_patches, w_patches, 299, 299, 3))

    for i in range(h_patches):
        for j in range(w_patches):
            patches[i, j] = img[i*299:(i+1)*299, j*299:(j+1)*299]

    patches = densenet.preprocess_input(patches.reshape((-1, 299, 299, 3)))
    pred_patch = model.predict(patches, verbose=0)

    return np.mean(pred_patch, axis=0)  # mean feature vector of all patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with image_name and score")
    parser.add_argument("--save_path", type=str, default="features.mat", help="Output .mat file path")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#cpu only

    base_model = load_model(args.model)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Gather all supported image formats
    image_paths = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg")) +
        glob.glob(os.path.join(args.image_dir, "*.jpeg")) +
        glob.glob(os.path.join(args.image_dir, "*.png"))
    )
    feats_list = []
    names_list = []

    print("📦 Loading MOS values...")
    df = pd.read_csv(args.csv)
    df_dict = dict(zip(df['image_name'], df['score']))
    mos_list = []

    print("🔍 Extracting features...")
    for path in tqdm(image_paths, desc="Processing"):
        basename = os.path.basename(path)
        if basename not in df_dict:
            print(f"❗ Warning: {basename} not found in MOS CSV. Skipping.")
            continue
        feats = test_image(model, path)
        if feats is not None:
            feats_list.append(feats)
            mos_list.append(df_dict[basename])
            names_list.append(basename)

    scipy.io.savemat(args.save_path, {
        "feats_mat": np.asarray(feats_list, dtype=np.float32),
        "mos": np.asarray(mos_list, dtype=np.float32),
        "image_names": np.array(names_list)
    })

    print(f"✅ Features, MOS, and image names saved to {args.save_path}")