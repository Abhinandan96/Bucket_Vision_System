import os
import re
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


# ============================================================
# FOLDER CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Your bucket images are directly inside D:\Bucket_Scanning
IMAGE_FOLDER = BASE_DIR


# Supported image formats
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# SIFT VISION SYSTEM
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=1500
)

matcher = cv2.BFMatcher(
    cv2.NORM_L2
)


# ============================================================
# GET IMAGE NAME
# ============================================================

def get_bucket_name(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    name = re.sub(
        r"[_-]+",
        " ",
        name
    )

    return name


# ============================================================
# FIND ALL BUCKET IMAGES
# ============================================================

def get_bucket_images():

    images = []

    for filename in os.listdir(IMAGE_FOLDER):

        extension = os.path.splitext(
            filename
        )[1].lower()

        # Ignore demo.py and other files
        if extension not in IMAGE_EXTENSIONS:
            continue

        images.append({
            "filename": filename,
            "label": get_bucket_name(filename)
        })

    return sorted(
        images,
        key=lambda x: x["filename"]
    )


# ============================================================
# IMAGE COMPARISON
# ============================================================

def compare_images(
    reference_image,
    camera_image
):

    # Convert reference image to grayscale
    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # Convert camera image to grayscale
    camera_gray = cv2.cvtColor(
        camera_image,
        cv2.COLOR_BGR2GRAY
    )


    # Slight blur to reduce noise
    reference_gray = cv2.GaussianBlur(
        reference_gray,
        (3, 3),
        0
    )

    camera_gray = cv2.GaussianBlur(
        camera_gray,
        (3, 3),
        0
    )


    # ========================================================
    # FEATURE DETECTION
    # ========================================================

    keypoints1, descriptors1 = (
        sift.detectAndCompute(
            reference_gray,
            None
        )
    )


    keypoints2, descriptors2 = (
        sift.detectAndCompute(
            camera_gray,
            None
        )
    )


    if (
        descriptors1 is None
        or
        descriptors2 is None
    ):

        return {
            "match": False,
            "score": 0,
            "good_matches": 0,
            "inliers": 0
        }


    # ========================================================
    # FEATURE MATCHING
    # ========================================================

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )


    good_matches = []


    # Lowe ratio test
    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:

            good_matches.append(m)


    # ========================================================
    # RANSAC GEOMETRIC CHECK
    # ========================================================

    inliers = 0


    if len(good_matches) >= 8:

        source_points = np.float32([

            keypoints1[m.queryIdx].pt

            for m in good_matches

        ]).reshape(
            -1,
            1,
            2
        )


        destination_points = np.float32([

            keypoints2[m.trainIdx].pt

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

                5.0

            )

            if mask is not None:

                inliers = int(
                    mask.ravel().sum()
                )

        except:

            inliers = 0


    # ========================================================
    # MATCH SCORE
    # ========================================================

    geometry_ratio = (

        inliers /

        max(
            len(good_matches),
            1
        )

    )


    score = min(

        100.0,

        8.0 * inliers
        +
        25.0 * geometry_ratio

    )


    # ========================================================
    # DECISION
    # ========================================================

    # Initial feasibility threshold
    match = (

        inliers >= 10

        and

        score >= 55

    )


    return {

        "match": bool(match),

        "score": round(
            score,
            1
        ),

        "good_matches":
            len(good_matches),

        "inliers":
            inliers

    }


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def home():

    bucket_images = (
        get_bucket_images()
    )


    print(
        f"Found {len(bucket_images)} bucket images."
    )


    return render_template(

        "index.html",

        references=bucket_images

    )


# ============================================================
# CAMERA IMAGE API
# ============================================================

@app.route(
    "/api/check",
    methods=["POST"]
)
def check_bucket():

    expected = request.form.get(
        "expected",
        ""
    )


    frame = request.files.get(
        "frame"
    )


    # --------------------------------------------------------
    # CHECK EXPECTED BUCKET
    # --------------------------------------------------------

    if not expected:

        return jsonify({

            "ok": False,

            "error":
                "Please select an expected bucket."

        }), 400


    # --------------------------------------------------------
    # CHECK CAMERA FRAME
    # --------------------------------------------------------

    if frame is None:

        return jsonify({

            "ok": False,

            "error":
                "No camera image received."

        }), 400


    # Prevent path traversal
    expected = os.path.basename(
        expected
    )


    reference_path = os.path.join(

        IMAGE_FOLDER,

        expected

    )


    if not os.path.isfile(
        reference_path
    ):

        return jsonify({

            "ok": False,

            "error":
                "Reference image not found."

        }), 404


    # --------------------------------------------------------
    # READ CAMERA IMAGE
    # --------------------------------------------------------

    image_bytes = np.frombuffer(

        frame.read(),

        np.uint8

    )


    camera_image = cv2.imdecode(

        image_bytes,

        cv2.IMREAD_COLOR

    )


    # Read reference image
    reference_image = cv2.imread(

        reference_path

    )


    if camera_image is None:

        return jsonify({

            "ok": False,

            "error":
                "Camera image could not be read."

        }), 400


    if reference_image is None:

        return jsonify({

            "ok": False,

            "error":
                "Reference image could not be read."

        }), 400


    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    result = compare_images(

        reference_image,

        camera_image

    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    result["expected"] = (
        get_bucket_name(expected)
    )


    result["decision"] = (

        "CORRECT"

        if result["match"]

        else

        "REJECTED"

    )


    result["ok"] = True


    return jsonify(result)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return "OK"


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print(" BERGER BUCKET VISION SYSTEM")
    print("=" * 55)

    bucket_images = (
        get_bucket_images()
    )

    print(
        f"Bucket images found: {len(bucket_images)}"
    )

    for image in bucket_images:

        print(
            "  ✓",
            image["filename"]
        )

    print("=" * 55)
    print()

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )