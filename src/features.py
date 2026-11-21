# features.py
# ------------------------------------------------------------------
# Turns one fruit photo into 42 numbers (features).
#
# The SAME function, extract_features(), is used everywhere:
#   - to build data/features.csv        (python src/features.py)
#   - in predict.py for a new photo
#   - in the dashboard
#
# Steps for every photo:
#   1. resize it to 512 x 512
#   2. remove the background (keep only the fruit)
#   3. measure colour, texture, shape and lesion features on the fruit
# ------------------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

IMAGE_SIZE = 512

# The 42 features, in the order they appear in features.csv
FEATURE_NAMES = [
    # colour: average and spread of Red, Green, Blue
    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
    # colour: Hue, Saturation, Value (Value = brightness)
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    # colour: L*a*b* (L = lightness, a = green-red, b = blue-yellow)
    "lab_l_mean", "lab_l_std", "lab_a_mean", "lab_a_std", "lab_b_mean", "lab_b_std",
    # texture: GLCM
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    # texture: LBP histogram (10 bins)
    "lbp_0", "lbp_1", "lbp_2", "lbp_3", "lbp_4", "lbp_5", "lbp_6", "lbp_7", "lbp_8", "lbp_9",
    # texture: edges
    "edge_density",
    # shape
    "area_ratio", "perimeter", "circularity", "eccentricity",
    # lesion-specific
    "brown_dark_pct", "pale_pct", "dark_spot_pct", "n_dark_spots",
]


def load_image(image_path):
    """Read a photo and resize it to 512 x 512."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read image: " + str(image_path))
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image


def remove_background(image):
    """
    Returns a mask: 255 where the fruit is, 0 where the background is.

    Idea: fruit skin is strongly coloured (high saturation). Paper, shadows
    and black borders are grey / white / black (low saturation).
    So we threshold the saturation channel.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    brightness = hsv[:, :, 2]

    # Otsu's method picks the threshold automatically
    threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep the threshold in a sensible range (matters when the fruit fills the whole photo)
    threshold = np.clip(threshold, 30, 90)

    # fruit = colourful AND not almost black
    mask = (saturation > threshold) & (brightness > 25)
    mask = mask.astype(np.uint8) * 255

    # clean up small specks and small gaps
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_kernel)

    # keep only the largest connected region (that is the fruit)
    count, regions, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        # nothing found, so use the whole image
        return np.full(mask.shape, 255, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]          # row 0 is the background, skip it
    largest = 1 + np.argmax(areas)
    mask = (regions == largest).astype(np.uint8) * 255

    # a fruit is round, so we take the convex hull of the region.
    # This puts back shaded skin, glare and lesions that the threshold missed.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(biggest_contour)
    fruit_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(fruit_mask, hull, 255)

    # if the mask is tiny something went wrong, so use the whole image
    if fruit_mask.mean() / 255 < 0.02:
        return np.full(mask.shape, 255, np.uint8)
    return fruit_mask


def extract_features(image_path):
    """Photo path -> dictionary with the 42 features."""
    image = load_image(image_path)
    mask = remove_background(image)
    fruit = mask > 0                      # True for fruit pixels

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # "inside" = the fruit without its outer rim, so the fruit outline
    # is not counted as an edge or a dark spot
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inside = cv2.erode(mask, rim_kernel) > 0
    if inside.sum() < 100:
        inside = fruit

    features = {}

    # ---------------- 1. COLOUR ----------------
    # mean and standard deviation of every colour channel, fruit pixels only
    # (OpenCV stores colour images as Blue, Green, Red)
    channels = {
        "r": image[:, :, 2], "g": image[:, :, 1], "b": image[:, :, 0],
        "h": hsv[:, :, 0], "s": hsv[:, :, 1], "v": hsv[:, :, 2],
        "lab_l": lab[:, :, 0], "lab_a": lab[:, :, 1], "lab_b": lab[:, :, 2],
    }
    for name in channels:
        pixels = channels[name][fruit].astype(np.float64)
        features[name + "_mean"] = pixels.mean()
        features[name + "_std"] = pixels.std()

    # ---------------- 2. TEXTURE: GLCM ----------------
    # GLCM counts how often two grey levels sit next to each other.
    # We reduce the image to 32 grey levels numbered 1..32 and give the
    # background the number 0. After building the matrix we delete row 0
    # and column 0, so pairs that touch the background are ignored.
    levels = (gray.astype(np.float64) / 256 * 32).astype(np.uint8) + 1
    levels[~fruit] = 0
    glcm = graycomatrix(levels, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=33, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    totals[totals == 0] = 1
    glcm = glcm / totals                                  # turn counts into probabilities
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        features["glcm_" + prop] = graycoprops(glcm, prop).mean()     # average of the 4 directions

    # ---------------- 3. TEXTURE: LBP ----------------
    # LBP compares each pixel with its 8 neighbours and gives a pattern number 0..9.
    # The histogram says how common each pattern is on the fruit.
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    counts, _ = np.histogram(lbp[inside], bins=np.arange(11))
    for i in range(10):
        features["lbp_" + str(i)] = counts[i] / max(counts.sum(), 1)

    # ---------------- 4. TEXTURE: EDGES ----------------
    # share of fruit pixels that are edges (lesion borders create edges)
    edges = cv2.Canny(gray, 100, 200) > 0
    features["edge_density"] = (edges & inside).sum() / inside.sum()

    # ---------------- 5. SHAPE ----------------
    region = regionprops(label(fruit))[0]         # the mask is one region: the fruit
    area = float(region.area)
    perimeter = float(region.perimeter)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    features["area_ratio"] = area / mask.size             # how much of the photo is fruit
    features["perimeter"] = perimeter
    features["circularity"] = min(circularity, 1.0)       # 1.0 = perfect circle
    features["eccentricity"] = region.eccentricity        # 0 = circle, near 1 = stretched

    # ---------------- 6. LESION-SPECIFIC ----------------
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1].astype(np.float64)
    val = hsv[:, :, 2].astype(np.float64)
    n_inside = max(inside.sum(), 1)
    typical_brightness = np.median(val[inside])
    typical_saturation = np.median(sat[inside])

    # brown pixels that are darker than the normal skin of this fruit
    brown_dark = inside & (hue >= 5) & (hue <= 30) & (sat >= 40) & (val < 0.7 * typical_brightness)
    features["brown_dark_pct"] = 100.0 * brown_dark.sum() / n_inside

    # pale / greyish pixels (corky lesion tissue loses its colour)
    pale = inside & (sat < 0.5 * typical_saturation)
    features["pale_pct"] = 100.0 * pale.sum() / n_inside

    # small dark spots: the "black-hat" operation makes small dark things bright
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    spot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, spot_kernel)
    spots = ((blackhat > 25) & inside).astype(np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    features["dark_spot_pct"] = 100.0 * spots.sum() / n_inside

    # count the spots that are at least 12 pixels big
    count, _, stats, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
    n_spots = 0
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            n_spots += 1
    features["n_dark_spots"] = float(n_spots)

    # return the features as plain numbers, in the fixed order
    ordered = {}
    for name in FEATURE_NAMES:
        ordered[name] = float(features[name])
    return ordered


# ------------------------------------------------------------------
# Run this file to build data/features.csv from all images
# ------------------------------------------------------------------
if __name__ == "__main__":
    metadata = pd.read_csv("data/metadata.csv")
    rows = []
    for i in range(len(metadata)):
        info = metadata.iloc[i]
        row = {"image_id": info["image_id"], "sample_id": info["sample_id"], "label": info["label"]}
        row.update(extract_features("data/raw/" + info["filename"]))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(i + 1, "of", len(metadata), "images done")

    features_table = pd.DataFrame(rows)
    features_table.to_csv("data/features.csv", index=False)
    print("Saved data/features.csv:", len(features_table), "images,", len(FEATURE_NAMES), "features")
    print(features_table["label"].value_counts())
