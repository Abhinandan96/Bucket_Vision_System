import os
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
# VISION ENGINE
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
# FIND ALL PRODUCT IMAGES
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
# COLOR HISTOGRAM
# ============================================================

def color_histogram(image):

    height, width = image.shape[:2]

    # Use central portion of image
    y1 = int(height * 0.14)
    y2 = int(height * 0.86)

    x1 = int(width * 0.14)
    x2 = int(width * 0.86)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        crop = image

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
# COMPARE ONE PRODUCT WITH CAMERA IMAGE
# ============================================================

def match_reference(
    reference,
    camera_image
):

    if reference is None:
        return 0.0, 0, 0, 0.0

    if reference["descriptors"] is None:
        return 0.0, 0, 0, 0.0

    # --------------------------------------------------------
    # CAMERA IMAGE
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

    camera_keypoints, camera_descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    if camera_descriptors is None:
        return 0.0, 0, 0, 0.0

    if len(camera_descriptors) < 2:
        return 0.0, 0, 0, 0.0

    # --------------------------------------------------------
    # KNN MATCHING
    # --------------------------------------------------------

    try:

        matches = matcher.knnMatch(
            reference["descriptors"],
            camera_descriptors,
            k=2
        )

    except Exception:

        return 0.0, 0, 0, 0.0

    good_matches = []

    # Strict ratio test
    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.68 * n.distance:

            good_matches.append(m)

    # --------------------------------------------------------
    # NOT ENOUGH MATCHES
    # --------------------------------------------------------

    if len(good_matches) < 8:

        return (
            0.0,
            len(good_matches),
            0,
            0.0
        )

    # --------------------------------------------------------
    # HOMOGRAPHY / RANSAC
    # --------------------------------------------------------

    source_points = np.float32([
        reference["keypoints"][m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    destination_points = np.float32([
        camera_keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    try:

        homography, mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            4.0
        )

    except Exception:

        mask = None

    if mask is None:

        inliers = 0

    else:

        inliers = int(mask.sum())

    # --------------------------------------------------------
    # GEOMETRIC RATIO
    # --------------------------------------------------------

    geometry_ratio = (
        inliers /
        max(len(good_matches), 1)
    )

    # --------------------------------------------------------
    # COLOR SIMILARITY
    # --------------------------------------------------------

    try:

        camera_hist = color_histogram(
            camera_image
        )

        correlation = cv2.compareHist(
            reference["hist"],
            camera_hist,
            cv2.HISTCMP_CORREL
        )

        color_score = float(
            np.clip(
                (correlation + 1.0) * 50.0,
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
        inliers * 4.0
        + geometry_ratio * 50.0
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    total_score = (
        feature_score * 0.80
        + color_score * 0.20
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

    path = os.path.join(
        IMAGE_FOLDER,
        item["filename"]
    )

    prepared = prepare_reference(path)

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
# DECODE UPLOADED CAMERA IMAGE
# ============================================================

def decode_uploaded_image():

    # --------------------------------------------------------
    # METHOD 1:
    # Normal multipart file upload
    # --------------------------------------------------------

    if "frame" in request.files:

        uploaded_file = request.files["frame"]

        image_bytes = uploaded_file.read()

        if not image_bytes:

            return None

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        if image_array.size == 0:

            return None

        try:

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

        except Exception:

            return None

        return image

    # --------------------------------------------------------
    # METHOD 2:
    # Fallback for Base64 clients
    # --------------------------------------------------------

    frame_data = request.form.get(
        "frame",
        ""
    )

    if not frame_data:

        return None

    try:

        import base64

        if "," in frame_data:

            frame_data = frame_data.split(
                ",",
                1
            )[1]

        frame_data = "".join(
            frame_data.split()
        )

        image_bytes = base64.b64decode(
            frame_data + "=" * (
                (-len(frame_data)) % 4
            ),
            validate=False
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


# ============================================================
# MAIN CAMERA CHECK
# ============================================================

@app.route(
    "/api/check",
    methods=["POST"]
)
def check():

    # --------------------------------------------------------
    # Expected product
    # --------------------------------------------------------

    expected = request.form.get(
        "expected",
        ""
    ).strip()

    if expected not in REFERENCE_DATA:

        return jsonify({
            "error": "Invalid expected bucket"
        }), 400

    # --------------------------------------------------------
    # Decode camera image
    # --------------------------------------------------------

    camera_image = decode_uploaded_image()

    if camera_image is None:

        return jsonify({
            "error": "Could not decode camera image"
        }), 400

    # --------------------------------------------------------
    # Check image dimensions
    # --------------------------------------------------------

    if (
        camera_image.shape[0] < 50
        or camera_image.shape[1] < 50
    ):

        return jsonify({
            "error": "Camera image is too small"
        }), 400

    # ========================================================
    # CLASSIFY AGAINST ALL PRODUCTS
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
    # Sort by score
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if not results:

        return jsonify({
            "error": "No reference images available"
        }), 500

    # --------------------------------------------------------
    # BEST PRODUCT
    # --------------------------------------------------------

    best = results[0]

    second = (
        results[1]
        if len(results) > 1
        else None
    )

    # --------------------------------------------------------
    # EXPECTED PRODUCT RESULT
    # --------------------------------------------------------

    expected_result = next(
        (
            r for r in results
            if r["filename"] == expected
        ),
        None
    )

    if expected_result is None:

        return jsonify({
            "error": "Expected product not found"
        }), 500

    # ========================================================
    # DECISION
    # ========================================================

    # Minimum confidence
    confidence_ok = (
        best["score"] >= 50
        and best["inliers"] >= 10
    )

    # Difference between best and second-best
    if second is None:

        margin_ok = True

    else:

        margin_ok = (
            best["score"]
            - second["score"]
            >= 12
        )

    # MOST IMPORTANT CONDITION
    #
    # Actual detected product MUST equal expected product

    expected_is_best = (
        best["filename"]
        == expected
    )

    is_match = (
        confidence_ok
        and margin_ok
        and expected_is_best
    )

    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if is_match:

        decision = "CORRECT"

        message = (
            "Bucket matches: "
            + best["label"]
        )

    else:

        decision = "REJECTED"

        message = (
            "Wrong bucket. Expected: "
            + get_bucket_name(expected)
            + ". Detected: "
            + best["label"]
        )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return jsonify({

        "match": is_match,

        "decision": decision,

        "expected": get_bucket_name(
            expected
        ),

        "actual": best["label"],

        "classifier_winner": best["label"],

        "message": message,

        "score": round(
            expected_result["score"],
            1
        ),

        "good_matches": expected_result[
            "good_matches"
        ],

        "inliers": expected_result[
            "inliers"
        ],

        "best_score": round(
            best["score"],
            1
        ),

        "second_score": round(
            second["score"],
            1
        ) if second else 0
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return "OK", 200


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    print(
        f"Found {len(BUCKET_IMAGES)} "
        "bucket images."
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
