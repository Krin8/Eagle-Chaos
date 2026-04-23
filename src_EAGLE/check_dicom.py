import os
import pydicom

DATASET_PATH = r"D:\Chaos\EAGLE\src_EAGLE\pytorch_data_dir\archive\CHAOS_Train_Sets"   # 👈 change if needed

bad_files = []

for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        path = os.path.join(root, f)
        try:
            pydicom.dcmread(path)
        except Exception:
            bad_files.append(path)

print(f"\nFound {len(bad_files)} bad files:\n")
for f in bad_files[:20]:   # print first 20
    print(f)

print("\nDone.")