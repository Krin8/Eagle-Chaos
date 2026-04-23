# =========================================
# IMPORTS
# =========================================
import os
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import random

from torch.utils.data import DataLoader, Dataset
from train_segmentation_eigen import LitUnsupervisedSegmenter
from modules import *
from utils import *
from crf import dense_crf
import torchvision.transforms as T

# =========================================
# DEVICE SETUP
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================
# PATHS
# =========================================
MODEL_PATH = "/content/drive/.shortcut-targets-by-id/1MxE7v7JrMeInzuP5EZtjusOZ1WOjuE-6/EAGLE/checkpoints/chaos/chaos_Apr09_20-09-06_vit_small_chaos_EAGLE/epoch=01-step=00002100-test/cluster/mIoU=13.89.ckpt"

DICOM_ROOT = "/content/drive/MyDrive/EAGLE/src_EAGLE/pytorch_data_dir/archive/CHAOS_Test_Sets/Test_Sets/CT"
OUTPUT_IMG_DIR = "/content/drive/MyDrive/EAGLE/images"

RES = 224
BATCH_SIZE = 4
NUM_SAMPLES_TO_SHOW = 5
EXPECTED_IMAGES = 1000   # adjust if needed

# =========================================
# STEP 1: CONVERT DICOM → PNG (ONLY ONCE)
# =========================================
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

existing_pngs = [f for f in os.listdir(OUTPUT_IMG_DIR) if f.endswith(".png")]

if len(existing_pngs) >= EXPECTED_IMAGES:
    print(f"✅ PNGs already exist ({len(existing_pngs)}). Skipping conversion.")
else:
    print("🔄 Converting DICOM → PNG (one-time)...")

    count = 0

    for patient in os.listdir(DICOM_ROOT):
        patient_path = os.path.join(DICOM_ROOT, patient)

        if not os.path.isdir(patient_path):
            continue

        dicom_path = os.path.join(patient_path, "DICOM_anon")
        if not os.path.exists(dicom_path):
            continue

        for file in os.listdir(dicom_path):
            if file.endswith(".dcm"):
                dcm = pydicom.dcmread(os.path.join(dicom_path, file))
                img = dcm.pixel_array.astype(np.float32)

                img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                img = (img * 255).astype(np.uint8)

                Image.fromarray(img).save(f"{OUTPUT_IMG_DIR}/ct_{count}.png")
                count += 1

    print(f"✅ Converted {count} images")

# =========================================
# STEP 2: DATASET
# =========================================
class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = [f for f in os.listdir(root) if f.endswith(".png")]

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.images[idx])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

transform = T.Compose([
    T.Resize((RES, RES)),
    T.ToTensor(),
    normalize
])

dataset = SimpleDataset(OUTPUT_IMG_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("✅ Dataset size:", len(dataset))

# =========================================
# STEP 3: LOAD MODEL
# =========================================
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

model = LitUnsupervisedSegmenter(**checkpoint['hyper_parameters'])
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device).eval()

print("✅ Model loaded")

# =========================================
# HELPER: COLORIZE
# =========================================
def colorize(mask):
    cmap = plt.cm.get_cmap('tab20')
    colored = cmap(mask / (mask.max() + 1e-5))
    return (colored[:, :, :3] * 255).astype(np.uint8)

# =========================================
# STEP 4: INFERENCE
# =========================================
results = []

for img in tqdm(loader):
    img = img.to(device)

    with torch.no_grad():
        feats, code1 = model.net(img)
        feats, code2 = model.net(img.flip(dims=[3]))

        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        for j in range(img.shape[0]):
            single_img = img[j].cpu()

            linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
            cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

            results.append({
                "img": single_img,
                "linear": linear_crf,
                "cluster": cluster_crf
            })

# =========================================
# STEP 5: VISUALIZATION
# =========================================
samples = random.sample(results, min(NUM_SAMPLES_TO_SHOW, len(results)))

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))

if len(samples) == 1:
    axes = [axes]

for i, sample in enumerate(samples):
    img = sample["img"].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    axes[i][0].imshow(img)
    axes[i][0].set_title("Original CT")

    axes[i][1].imshow(colorize(sample["cluster"]))
    axes[i][1].set_title("Cluster Segmentation")

    axes[i][2].imshow(colorize(sample["linear"]))
    axes[i][2].set_title("Linear Segmentation")

    for j in range(3):
        axes[i][j].axis("off")

plt.tight_layout()
plt.show()