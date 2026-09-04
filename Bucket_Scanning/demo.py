import os
import base64
import binascii

import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template


# ============================================================
# SETTINGS
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
# SIFT + MATCHER
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=2000
)

matcher = cv2.BFMatcher(
    cv2.NORM_L2
)


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

        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):

            images.append({
                "filename": filename,
                "label": get_bucket_name(filename)
            })

    return images


# ============================================================
# CENTER CROP
# ============================================================

def center_crop(image, fraction=0.72):

    height, width = image.shape[:2]

    crop_height = int(
        height * fraction
    )

    crop_width = int(
        width * fraction
    )

    y1 = max(
        0,
        (height - crop_height) // 2
    )

    x1 = max(
        0,
        (width - crop_width) // 2
    )

    return image[
        y1:y1 + crop_height,
        x1:x1 + crop_width
    ]


# ============================================================
# COLOR HISTOGRAM
# ============================================================

def color_histogram(image):

    crop = center_crop(
        image,
        0.72
    )

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

    image = cv2.imread(
        path
    )

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

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    return {
        "image": image,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "hist": color_histogram(image)
    }


# ============================================================
# MATCH ONE REFERENCE AGAINST CAMERA IMAGE
# ============================================================

def match_reference(
    reference,
    camera_image
):

    if reference is None:

        return (
            0.0,
            0,
            0,
            0.0
        )

    if reference["descriptors"] is None:

        return (
            0.0,
            0,
            0,
            0.0
        )

    # --------------------------------------------------------
    # CAMERA IMAGE → GRAYSCALE
    # --------------------------------------------------------

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

    if descriptors_camera is None:

        return (
            0.0,
            0,
            0,
            0.0
        )

    if len(descriptors_camera) < 2:

        return (
            0.0,
            0,
            0,
            0.0
        )

    # --------------------------------------------------------
    # KNN FEATURE MATCHING
    # --------------------------------------------------------

    try:

        matches = matcher.knnMatch(
            reference["descriptors"],
            descriptors_camera,
            k=2
        )

    except Exception:

        return (
            0.0,
            0,
            0,
            0.0
        )

    good_matches = []

    # Stricter Lowe ratio
    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.68 * n.distance:

            good_matches.append(m)

    # --------------------------------------------------------
    # NOT ENOUGH MATCHES
    # --------------------------------------------------------

    if len(good_matches) < 10:

        return (
            0.0,
            len(good_matches),
            0,
            0.0
        )

    # --------------------------------------------------------
    # RANSAC HOMOGRAPHY
    # --------------------------------------------------------

    source_points = np.float32([
        reference["keypoints"][
            m.queryIdx
        ].pt
        for m in good_matches
    ]).reshape(
        -1,
        1,
        2
    )

    destination_points = np.float32([
        keypoints_camera[
            m.trainIdx
        ].pt
        for m in good_matches
    ]).reshape(
        -1,
        1,
        2
    )

    try:

        homography, mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            4.0
        )

    except Exception:

        homography = None
        mask = None

    if mask is None:

        inliers = 0

    else:

        inliers = int(
            mask.sum()
        )

    # --------------------------------------------------------
    # GEOMETRIC QUALITY
    # --------------------------------------------------------

    geometry_ratio = (
        inliers /
        max(
            len(good_matches),
            1
        )
    )

    # --------------------------------------------------------
    # COLOR SIMILARITY
    # --------------------------------------------------------

    try:

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
                (
                    color_correlation
                    + 1.0
                )
                * 50.0,
                0.0,
                100.0
            )
        )

    except Exception:

        color_score = 0.0

    # --------------------------------------------------------
    # FEATURE SCORE
    # --------------------------------------------------------

    feature_score = min(
        100.0,
        (
            inliers * 4.0
            + geometry_ratio * 50.0
        )
    )

    # --------------------------------------------------------
    # FINAL SCORE
    #
    # Feature matching = 80%
    # Color similarity = 20%
    # --------------------------------------------------------

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
# LOAD ALL PRODUCT REFERENCE IMAGES
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
# CAMERA IMAGE DECODER
# ============================================================

def decode_camera_image(frame_data):

    if not frame_data:

        return None

    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    frame_data = str(
        frame_data
    ).strip()

    # --------------------------------------------------------
    # Remove DATA URL prefix
    #
    # Example:
    # data:image/jpeg;base64,/9j/4AAQ...
    # --------------------------------------------------------

    if "," in frame_data:

        first_part, second_part = (
            frame_data.split(
                ",",
                1
            )
        )

        # If it looks like a data URL,
        # use only the Base64 section.
        if "base64" in first_part.lower():

            frame_data = second_part

    # --------------------------------------------------------
    # Remove whitespace
    # --------------------------------------------------------

    frame_data = "".join(
        frame_data.split()
    )

    if not frame_data:

        return None

    # --------------------------------------------------------
    # Fix Base64 padding
    # --------------------------------------------------------

    remainder = len(frame_data) % 4

    if remainder != 0:

        frame_data += "=" * (
            4 - remainder
        )

    # --------------------------------------------------------
    # Decode Base64
    # --------------------------------------------------------

    try:

        image_bytes = base64.b64decode(
            frame_data,
            validate=False
        )

    except (
        ValueError,
        binascii.Error,
        TypeError
    ):

        # Try URL-safe Base64 as fallback
        try:

            image_bytes = base64.urlsafe_b64decode(
                frame_data
            )

        except Exception:

            return None

    if not image_bytes:

        return None

    # --------------------------------------------------------
    # Bytes → NumPy
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    if image_array.size == 0:

        return None

    # --------------------------------------------------------
    # NumPy → OpenCV image
    # --------------------------------------------------------

    try:

        camera_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

    except Exception:

        return None

    return camera_image


# ============================================================
# CAMERA CHECK API
# ============================================================

@app.route(
    "/api/check",
    methods=["POST"]
)
def check():

    # --------------------------------------------------------
    # Get expected bucket
    # --------------------------------------------------------

    expected = request.form.get(
        "expected",
        ""
    ).strip()

    # --------------------------------------------------------
    # Get camera frame
    # --------------------------------------------------------

    frame_data = request.form.get(
        "frame",
        ""
    )

    # --------------------------------------------------------
    # Validate expected product
    # --------------------------------------------------------

    if expected not in REFERENCE_DATA:

        return jsonify({
            "error": "Invalid expected bucket"
        }), 400

    # --------------------------------------------------------
    # Decode camera image
    # --------------------------------------------------------

    camera_image = decode_camera_image(
        frame_data
    )

    if camera_image is None:

        return jsonify({
            "error": "Could not decode camera image"
        }), 400

    # ========================================================
    # COMPARE CAMERA IMAGE WITH ALL PRODUCTS
    # ========================================================

    results = []

    for item in BUCKET_IMAGES:

        filename = item[
            "filename"
        ]

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

            "score": float(score),

            "good_matches": int(
                good_matches
            ),

            "inliers": int(
                inliers
            ),

            "color_score": float(
                color_score
            )
        })

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        return jsonify({
            "error": "No reference images available"
        }), 500

    # ========================================================
    # SORT BY BEST MATCH
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = results[0]

    if len(results) > 1:

        second = results[1]

    else:

        second = None

    # ========================================================
    # FIND EXPECTED PRODUCT RESULT
    # ========================================================

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

    # Condition 1:
    # Best product must have enough quality.

    absolute_ok = (
        best["score"] >= 50
        and best["inliers"] >= 10
    )

    # Condition 2:
    # Best product should clearly beat
    # the second-best product.

    if second is None:

        margin_ok = True

    else:

        margin_ok = (
            best["score"]
            - second["score"]
            >= 12
        )

    # Condition 3:
    # MOST IMPORTANT:
    #
    # The BEST CLASSIFIED PRODUCT must
    # be the same as the EXPECTED PRODUCT.

    expected_is_best = (
        best["filename"]
        == expected
    )

    # Final decision

    is_match = (
        absolute_ok
        and margin_ok
        and expected_is_best
    )

    # ========================================================
    # MESSAGE
    # ========================================================

    if is_match:

        decision = "CORRECT"

        message = (
            f"Bucket matches: "
            f"{best['label']}"
        )

    else:

        decision = "REJECTED"

        message = (
            f"Wrong bucket. "
            f"Expected: "
            f"{get_bucket_name(expected)}. "
            f"Detected: "
            f"{best['label']}"
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return jsonify({

        "match": is_match,

        "expected": get_bucket_name(
            expected
        ),

        "actual": best["label"],

        "classifier_winner": best["label"],

        "decision": decision,

        # Score for EXPECTED PRODUCT
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

        # Best classified product score
        "best_score": round(
            best["score"],
            1
        ),

        # Second-best product score
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
# LOCAL DEVELOPMENT
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
