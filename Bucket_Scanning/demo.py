import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template


# ============================================================
# BASIC SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = BASE_DIR

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

app = Flask(__name__)


# ============================================================
# VISION SETUP
# ============================================================

# SIFT detects visual features
sift = cv2.SIFT_create(
    nfeatures=2000
)

# SIFT uses L2 distance
matcher = cv2.BFMatcher(cv2.NORM_L2)


# ============================================================
# PRODUCT NAME
# ============================================================

def get_bucket_name(filename):

    name = os.path.splitext(filename)[0]

    return name.replace("_", " ").replace("-", " ")


# ============================================================
# FIND ALL BUCKET IMAGES
# ============================================================

def get_bucket_images():

    images = []

    for filename in sorted(os.listdir(IMAGE_FOLDER)):

        if filename.lower().endswith(SUPPORTED_EXTENSIONS):

            images.append({
                "filename": filename,
                "label": get_bucket_name(filename)
            })

    return images


# ============================================================
# CENTER CROP
# Helps reduce background influence
# ============================================================

def center_crop(image, fraction=0.72):

    height, width = image.shape[:2]

    crop_height = int(height * fraction)
    crop_width = int(width * fraction)

    y1 = max(0, (height - crop_height) // 2)
    x1 = max(0, (width - crop_width) // 2)

    return image[
        y1:y1 + crop_height,
        x1:x1 + crop_width
    ]


# ============================================================
# COLOR HISTOGRAM
# Used only as a supporting feature
# ============================================================

def color_histogram(image):

    crop = center_crop(image, 0.72)

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist,
        hist
    )

    return hist


# ============================================================
# PREPARE REFERENCE IMAGE
# ============================================================

def prepare_reference(path):

    image = cv2.imread(path)

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return {
        "image": image,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "hist": color_histogram(image)
    }


# ============================================================
# COMPARE CAMERA IMAGE WITH ONE REFERENCE
# ============================================================

def match_reference(reference, camera_image):

    if (
        reference is None
        or reference["descriptors"] is None
    ):
        return 0.0, 0, 0, 0.0

    # -------------------------------
    # Camera image → grayscale
    # -------------------------------

    gray = cv2.cvtColor(
        camera_image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    keypoints_camera, descriptors_camera = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    if (
        descriptors_camera is None
        or len(descriptors_camera) < 2
    ):
        return 0.0, 0, 0, 0.0

    # -------------------------------
    # Feature matching
    # -------------------------------

    matches = matcher.knnMatch(
        reference["descriptors"],
        descriptors_camera,
        k=2
    )

    good_matches = []

    # Stricter Lowe ratio than before
    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.68 * n.distance:

                good_matches.append(m)

    # Not enough reliable features
    if len(good_matches) < 10:

        return (
            0.0,
            len(good_matches),
            0,
            0.0
        )

    # -------------------------------
    # Homography / RANSAC
    # -------------------------------

    source_points = np.float32([
        reference["keypoints"][m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    destination_points = np.float32([
        keypoints_camera[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        4.0
    )

    if mask is None:

        inliers = 0

    else:

        inliers = int(mask.sum())

    # -------------------------------
    # Geometry quality
    # -------------------------------

    geometry_ratio = (
        inliers / max(len(good_matches), 1)
    )

    # -------------------------------
    # Color similarity
    # -------------------------------

    camera_hist = color_histogram(
        camera_image
    )

    color_correlation = cv2.compareHist(
        reference["hist"],
        camera_hist,
        cv2.HISTCMP_CORREL
    )

    color_score = float(
        np.clip(
            (color_correlation + 1.0) * 50.0,
            0.0,
            100.0
        )
    )

    # -------------------------------
    # Feature score
    # -------------------------------

    feature_score = min(
        100.0,
        inliers * 4.0
        + geometry_ratio * 50.0
    )

    # -------------------------------
    # Final score
    #
    # Feature matching = 80%
    # Color = 20%
    # -------------------------------

    total_score = (
        0.80 * feature_score
        + 0.20 * color_score
    )

    return (
        total_score,
        len(good_matches),
        inliers,
        color_score
    )


# ============================================================
# LOAD ALL 14 PRODUCTS
# ============================================================

BUCKET_IMAGES = get_bucket_images()

REFERENCE_DATA = {}

for item in BUCKET_IMAGES:

    image_path = os.path.join(
        IMAGE_FOLDER,
        item["filename"]
    )

    prepared = prepare_reference(
        image_path
    )

    if prepared is not None:

        REFERENCE_DATA[
            item["filename"]
        ] = prepared


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        references=BUCKET_IMAGES
    )


# ============================================================
# CAMERA CHECK
# ============================================================

@app.route(
    "/api/check",
    methods=["POST"]
)
def check():

    expected = request.form.get(
        "expected",
        ""
    )

    frame_data = request.form.get(
        "frame",
        ""
    )

    # -------------------------------
    # Validate expected product
    # -------------------------------

    if expected not in REFERENCE_DATA:

        return jsonify({
            "error": "Invalid expected bucket"
        }), 400

    # -------------------------------
    # Remove data URL header
    # -------------------------------

    if "," in frame_data:

        frame_data = frame_data.split(
            ",",
            1
        )[1]

    # -------------------------------
    # Decode camera image
    # -------------------------------

    try:

        image_bytes = np.frombuffer(
            base64.b64decode(frame_data),
            dtype=np.uint8
        )

        camera_image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

    except Exception:

        return jsonify({
            "error": "Could not decode camera image"
        }), 400

    if camera_image is None:

        return jsonify({
            "error": "Invalid camera image"
        }), 400

    # ========================================================
    # IMPORTANT:
    #
    # COMPARE CAMERA IMAGE WITH ALL PRODUCTS
    # ========================================================

    results = []

    for item in BUCKET_IMAGES:

        filename = item["filename"]

        if filename not in REFERENCE_DATA:

            continue

        (
            score,
            good_matches,
            inliers,
            color_score
        ) = match_reference(
            REFERENCE_DATA[filename],
            camera_image
        )

        results.append({

            "filename": filename,

            "label": item["label"],

            "score": score,

            "good_matches": good_matches,

            "inliers": inliers,

            "color_score": color_score
        })

    # -------------------------------
    # Sort highest score first
    # -------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if not results:

        return jsonify({
            "error": "No reference images available"
        }), 500

    # ========================================================
    # BEST DETECTED PRODUCT
    # ========================================================

    best = results[0]

    if len(results) > 1:

        second = results[1]

    else:

        second = None

    # Find the score of EXPECTED product
    expected_result = next(
        (
            result
            for result in results
            if result["filename"] == expected
        ),
        None
    )

    if expected_result is None:

        return jsonify({
            "error": "Expected product was not classified"
        }), 500

    # ========================================================
    # DECISION LOGIC
    # ========================================================

    # The best product must have:
    #
    # 1. Enough overall score
    # 2. Enough RANSAC inliers
    # 3. A clear lead over second-best
    # 4. MOST IMPORTANT:
    #    best product MUST equal expected product
    # ========================================================

    absolute_ok = (
        best["score"] >= 50
        and best["inliers"] >= 10
    )

    margin_ok = (
        second is None
        or (
            best["score"]
            - second["score"]
            >= 12
        )
    )

    expected_is_best = (
        best["filename"]
        == expected
    )

    is_match = (
        absolute_ok
        and margin_ok
        and expected_is_best
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    if is_match:

        message = (
            f"Bucket matches: "
            f"{best['label']}"
        )

        decision = "CORRECT"

    else:

        message = (
            f"Wrong bucket. "
            f"Expected: "
            f"{get_bucket_name(expected)}. "
            f"Detected: "
            f"{best['label']}"
        )

        decision = "REJECTED"

    return jsonify({

        "match": is_match,

        "expected": get_bucket_name(
            expected
        ),

        "actual": best["label"],

        "classifier_winner": best["label"],

        "decision": decision,

        # Score belonging to EXPECTED product
        "score": round(
            expected_result["score"],
            1
        ),

        "good_matches": (
            expected_result[
                "good_matches"
            ]
        ),

        "inliers": (
            expected_result[
                "inliers"
            ]
        ),

        # Useful for debugging
        "best_score": round(
            best["score"],
            1
        ),

        "second_score": round(
            second["score"],
            1
        ) if second else 0,

        "message": message
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return "OK", 200


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    print(
        f"Found {len(BUCKET_IMAGES)} "
        f"bucket images."
    )

    for item in BUCKET_IMAGES:

        print(
            " -",
            item["label"]
        )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
