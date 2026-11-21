# organize_dataset.py
# ------------------------------------------------------------------
# Dataset construction. This was run ONE time on the downloaded images.
# It is kept so we can show how the dataset was built.
#
#   python src/organize_dataset.py
#
# Before:  dataset/citrus-canker/citrus-canker (47).jpg
#          dataset/healthy/h(23).png
# After:   data/raw/C_S017_03.jpg     (C = Canker, sample 17, photo 3)
#          data/raw/H_S141_01.jpg     (H = Healthy, sample 141, photo 1)
#          data/metadata.csv          (one row per image)
#          data/excluded/             (removed files + excluded_log.csv)
#
# What it does:
#   1. removes unreadable files and exact duplicate files
#   2. finds images that show the SAME fruit and gives them the same sample_id
#      (downloaded datasets contain many rotated / flipped copies and several
#       photos of one fruit; they must stay together in train or in test)
#   3. removes fruits that are in BOTH class folders (we cannot trust the label)
#   4. renames the files and writes metadata.csv
# ------------------------------------------------------------------
import hashlib
import os
import shutil

import cv2
import numpy as np
import pandas as pd

from features import load_image, remove_background

if not os.path.exists("dataset"):
    print("dataset/ not found. The dataset is already organised in data/raw/. Nothing to do.")
    raise SystemExit

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/excluded", exist_ok=True)

# Found by looking at the matches by eye: this fruit is in both class folders,
# but the automatic check in step 3 does not always catch it.
ALSO_REMOVE = ["h(42).jpg", "citrus-canker (17).jpg", "citrus-canker (54).jpg", "citrus-canker (83).jpg"]


def number_in_name(filename):
    """ 'citrus-canker (47).jpg' -> 47, used to sort the files in a natural order. """
    digits = ""
    for character in filename:
        if character.isdigit():
            digits += character
    return int(digits)


# ------------------------------------------------------------------
# STEP 1: list all images, skip unreadable files and exact duplicates
# ------------------------------------------------------------------
images = []          # one dictionary per image we keep
excluded = []        # one dictionary per image we remove
seen_files = {}      # fingerprint of the file content -> first file that had it

for folder, label in [("citrus-canker", "Canker"), ("healthy", "Healthy")]:
    for filename in sorted(os.listdir("dataset/" + folder), key=number_in_name):
        path = "dataset/" + folder + "/" + filename

        if cv2.imread(path) is None:
            excluded.append({"path": path, "label": label, "reason": "unreadable / corrupted file"})
            continue

        # md5 is a fingerprint of the file: two identical files have the same md5
        with open(path, "rb") as f:
            fingerprint = hashlib.md5(f.read()).hexdigest()
        if fingerprint in seen_files:
            excluded.append({"path": path, "label": label,
                             "reason": "exact duplicate of " + seen_files[fingerprint]})
            continue
        seen_files[fingerprint] = filename

        if filename in ALSO_REMOVE:
            excluded.append({"path": path, "label": label,
                             "reason": "label conflict: same fruit is in both class folders (found by eye)"})
            continue

        images.append({"path": path, "filename": filename, "folder": folder, "label": label})

print("Images kept so far:", len(images), " removed:", len(excluded))

# ------------------------------------------------------------------
# STEP 2: find images of the same fruit
#
# SIFT finds "keypoints": small distinctive spots (a lesion, a pore pattern)
# and describes each one with numbers. If two photos share many keypoints,
# and those keypoints line up after ONE rotation + zoom + shift, the two
# photos show the same fruit.
# ------------------------------------------------------------------
sift = cv2.SIFT_create(nfeatures=600)
matcher = cv2.BFMatcher()
cv2.setRNGSeed(42)                       # so the result is the same every run


def find_keypoints(image_path, mirror):
    """Keypoints on the fruit only (not on the background). mirror=True flips the photo."""
    image = load_image(image_path)
    mask = remove_background(image)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))   # stay off the rim
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mirror:
        gray = cv2.flip(gray, 1)
        mask = cv2.flip(mask, 1)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    positions = np.float32([k.pt for k in keypoints]).reshape(-1, 2)
    return positions, descriptors


def count_matches(positions_a, descriptors_a, positions_b, descriptors_b):
    """How many keypoints of photo A are found in photo B at a consistent place."""
    if descriptors_a is None or descriptors_b is None:
        return 0
    if len(descriptors_a) < 8 or len(descriptors_b) < 8:
        return 0

    # for each keypoint in A, find its 2 most similar keypoints in B.
    # Keep it only if the best one is clearly better than the second best.
    good = []
    for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return 0

    # check that the matched points agree on ONE rotation + zoom + shift (RANSAC)
    points_a = np.float32([positions_a[m.queryIdx] for m in good])
    points_b = np.float32([positions_b[m.trainIdx] for m in good])
    transform, agrees = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC,
                                                    ransacReprojThreshold=5.0)
    if transform is None:
        return 0
    zoom = np.hypot(transform[0, 0], transform[0, 1])
    if zoom < 0.6 or zoom > 1.6:         # an unrealistic zoom means a false match
        return 0
    return int(agrees.sum())


print("Looking for images of the same fruit (takes about a minute)...")
normal = []
mirrored = []
for item in images:
    normal.append(find_keypoints(item["path"], mirror=False))
    mirrored.append(find_keypoints(item["path"], mirror=True))

# every image starts in its own group; when two images match, their groups are merged
n = len(images)
group = list(range(n))
for i in range(n):
    for j in range(i + 1, n):
        matches = max(count_matches(normal[i][0], normal[i][1], normal[j][0], normal[j][1]),
                      count_matches(normal[i][0], normal[i][1], mirrored[j][0], mirrored[j][1]))
        if matches >= 8 and group[i] != group[j]:
            old_group = group[j]
            for k in range(n):
                if group[k] == old_group:
                    group[k] = group[i]

# ------------------------------------------------------------------
# STEP 3: a fruit that is in BOTH class folders has an unreliable label -> remove it
# ------------------------------------------------------------------
kept_images = []
kept_groups = []
for i in range(n):
    labels_in_group = set()
    for k in range(n):
        if group[k] == group[i]:
            labels_in_group.add(images[k]["label"])
    if len(labels_in_group) > 1:
        excluded.append({"path": images[i]["path"], "label": images[i]["label"],
                         "reason": "label conflict: same fruit is in both class folders"})
    else:
        kept_images.append(images[i])
        kept_groups.append(group[i])

# ------------------------------------------------------------------
# STEP 4: give sample ids, rename the files, write metadata.csv
# ------------------------------------------------------------------
sample_id_of_group = {}      # group number -> "S001", "S002", ...
photos_in_sample = {}        # "S001" -> how many photos so far
rows = []

for item, g in zip(kept_images, kept_groups):
    if g not in sample_id_of_group:
        sample_id_of_group[g] = "S" + str(len(sample_id_of_group) + 1).zfill(3)
    sample_id = sample_id_of_group[g]
    photos_in_sample[sample_id] = photos_in_sample.get(sample_id, 0) + 1

    extension = os.path.splitext(item["filename"])[1].lower()
    if extension not in [".jpg", ".jpeg", ".png"]:
        extension = ".jpg"
    letter = "C" if item["label"] == "Canker" else "H"
    new_name = letter + "_" + sample_id + "_" + str(photos_in_sample[sample_id]).zfill(2) + extension

    height, width = cv2.imread(item["path"]).shape[:2]
    rows.append({
        "image_id": len(rows) + 1,
        "filename": new_name,
        "sample_id": sample_id,
        "label": item["label"],
        "source": "internet",
        "original_filename": item["folder"] + "/" + item["filename"],
        # session, date, photographer, lighting, distance and variety are NOT recorded:
        # the images are from the internet, so we do not know them and we do not guess
        "width": width, "height": height,
    })
    shutil.move(item["path"], "data/raw/" + new_name)

metadata = pd.DataFrame(rows)
metadata.to_csv("data/metadata.csv", index=False)

# move the removed files to data/excluded/ and write the log
log = []
for item in excluded:
    folder = os.path.basename(os.path.dirname(item["path"]))
    new_name = folder + "__" + os.path.basename(item["path"])
    shutil.move(item["path"], "data/excluded/" + new_name)
    log.append({"file": new_name, "label": item["label"], "reason": item["reason"]})
pd.DataFrame(log).to_csv("data/excluded/excluded_log.csv", index=False)

print("Saved", len(metadata), "images to data/raw/ and data/metadata.csv")
print("Images per class:")
print(metadata["label"].value_counts())
print("Fruits (sample ids) per class:")
print(metadata.groupby("label")["sample_id"].nunique())
print("Removed files moved to data/excluded/:", len(log))
