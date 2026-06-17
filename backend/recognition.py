from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCES_JSON = ROOT / "references" / "artworks.json"
INDEX_DIR = ROOT / "references" / "index"


@dataclass
class ArtworkReference:
    id: str
    title: str
    artist: str | None
    object_number: str | None
    image_path: Path
    keypoints: list
    descriptors: np.ndarray
    width: int
    height: int


@dataclass
class CandidateResult:
    artwork_id: str
    title: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    confidence: float
    accepted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "artworkId": self.artwork_id,
            "title": self.title,
            "goodMatches": self.good_matches,
            "inliers": self.inliers,
            "inlierRatio": round(self.inlier_ratio, 4),
            "confidence": round(self.confidence, 4),
            "accepted": self.accepted,
        }


class RecognitionError(Exception):
    pass


class ArtworkRecognizer:
    def __init__(
        self,
        references_json: Path = REFERENCES_JSON,
        index_dir: Path = INDEX_DIR,
        max_image_dimension: int = 1400,
        min_good_matches: int = 12,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.12,
        min_inlier_margin: int = 4,
        min_confidence_margin: float = 0.04,
    ) -> None:
        self.references_json = references_json
        self.index_dir = index_dir
        self.max_image_dimension = max_image_dimension
        self.min_good_matches = min_good_matches
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inlier_margin = min_inlier_margin
        self.min_confidence_margin = min_confidence_margin

        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        self.references: list[ArtworkReference] = []

        self.load_references()

    def _create_detector(self):
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(
                nfeatures=3500,
                contrastThreshold=0.03,
                edgeThreshold=10,
            )

        return cv2.ORB_create(nfeatures=4500)

    def _create_matcher(self):
        if hasattr(cv2, "SIFT_create"):
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=80)
            return cv2.FlannBasedMatcher(index_params, search_params)

        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @property
    def reference_count(self) -> int:
        return len(self.references)

    def load_references(self) -> None:
        self.references = []
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if not self.references_json.exists():
            raise RecognitionError(f"Missing {self.references_json}")

        data = json.loads(self.references_json.read_text(encoding="utf-8"))

        for item in data:
            image_path = (ROOT / item["image"]).resolve()

            if not image_path.exists():
                print(f"[RijksLens] Skipping missing reference image: {image_path}")
                continue

            try:
                reference = self._load_or_build_reference(item, image_path)
                self.references.append(reference)

                print(
                    f"[RijksLens] Loaded {reference.title}: "
                    f"{len(reference.keypoints)} keypoints"
                )
            except Exception as exc:
                print(f"[RijksLens] Failed to load {item.get('title')}: {exc}")

    def _load_or_build_reference(
        self,
        item: dict[str, Any],
        image_path: Path,
    ) -> ArtworkReference:
        reference_id = item["id"]
        cache_path = self.index_dir / f"{reference_id}.npz"
        image_stat = image_path.stat()

        if cache_path.exists():
            try:
                with np.load(str(cache_path), allow_pickle=True) as cached:
                    cached_mtime = float(cached["image_mtime"])

                    if abs(cached_mtime - image_stat.st_mtime) < 0.001:
                        keypoint_array = cached["keypoints"]
                        descriptors = cached["descriptors"]
                        height = int(cached["height"])
                        width = int(cached["width"])
                        keypoints = [array_to_keypoint(row) for row in keypoint_array]

                        return ArtworkReference(
                            id=reference_id,
                            title=item["title"],
                            artist=item.get("artist"),
                            object_number=item.get("objectNumber"),
                            image_path=image_path,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            width=width,
                            height=height,
                        )
            except Exception:
                pass

        gray = load_image_gray(image_path, self.max_image_dimension)
        keypoints, descriptors = self.extract_features(gray)

        if descriptors is None or len(keypoints) < 20:
            raise RecognitionError(f"Not enough visual features in {image_path}")

        keypoint_array = np.array(
            [keypoint_to_array(kp) for kp in keypoints],
            dtype=np.float32,
        )

        np.savez_compressed(
            str(cache_path),
            keypoints=keypoint_array,
            descriptors=descriptors,
            width=gray.shape[1],
            height=gray.shape[0],
            image_mtime=image_stat.st_mtime,
        )

        return ArtworkReference(
            id=reference_id,
            title=item["title"],
            artist=item.get("artist"),
            object_number=item.get("objectNumber"),
            image_path=image_path,
            keypoints=keypoints,
            descriptors=descriptors,
            width=gray.shape[1],
            height=gray.shape[0],
        )

    def extract_features(self, gray: np.ndarray):
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None:
            return [], None

        if descriptors.dtype != np.float32 and hasattr(cv2, "SIFT_create"):
            descriptors = descriptors.astype(np.float32)

        return keypoints, descriptors

    def recognize_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.references:
            raise RecognitionError("No reference artworks are loaded.")

        query_gray = decode_image_gray(image_bytes, self.max_image_dimension)
        query_keypoints, query_descriptors = self.extract_features(query_gray)

        if query_descriptors is None or len(query_keypoints) < 10:
            return {
                "accepted": False,
                "artworkId": None,
                "title": None,
                "message": "Not enough visual features in the uploaded photo.",
                "queryKeypoints": len(query_keypoints),
                "candidates": [],
            }

        candidates = [
            self.compare_to_reference(query_keypoints, query_descriptors, reference)
            for reference in self.references
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.inliers,
                candidate.confidence,
                candidate.good_matches,
            ),
            reverse=True,
        )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        second_inliers = second.inliers if second else 0
        second_confidence = second.confidence if second else 0.0

        inlier_margin = best.inliers - second_inliers
        confidence_margin = best.confidence - second_confidence

        accepted = (
            best.good_matches >= self.min_good_matches
            and best.inliers >= self.min_inliers
            and best.inlier_ratio >= self.min_inlier_ratio
            and (
                not second
                or inlier_margin >= self.min_inlier_margin
                or confidence_margin >= self.min_confidence_margin
            )
        )

        best.accepted = accepted

        if accepted:
            message = f"Recognized {best.title} with {best.inliers} geometric inliers."
        else:
            message = (
                f"No confident match. Best was {best.title} with "
                f"{best.inliers} inliers, {best.good_matches} good matches, "
                f"inlier ratio {best.inlier_ratio:.2f}."
            )

        return {
            "accepted": accepted,
            "artworkId": best.artwork_id if accepted else None,
            "title": best.title if accepted else None,
            "message": message,
            "queryKeypoints": len(query_keypoints),
            "bestCandidate": best.as_dict(),
            "secondCandidate": second.as_dict() if second else None,
            "inlierMargin": inlier_margin,
            "confidenceMargin": round(confidence_margin, 4),
            "candidates": [candidate.as_dict() for candidate in candidates],
        }

    def compare_to_reference(
        self,
        query_keypoints: list,
        query_descriptors: np.ndarray,
        reference: ArtworkReference,
    ) -> CandidateResult:
        if query_descriptors is None or reference.descriptors is None:
            return CandidateResult(reference.id, reference.title, 0, 0, 0.0, 0.0)

        try:
            raw_matches = self.matcher.knnMatch(
                query_descriptors,
                reference.descriptors,
                k=2,
            )
        except cv2.error:
            raw_matches = []

        good_matches = []
        ratio = 0.82

        for pair in raw_matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return CandidateResult(
                reference.id,
                reference.title,
                len(good_matches),
                0,
                0.0,
                0.0,
            )

        src_pts = np.float32(
            [query_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [reference.keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        _matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if mask is None:
            inliers = 0
        else:
            inliers = int(mask.ravel().sum())

        inlier_ratio = inliers / max(1, len(good_matches))
        confidence = compute_confidence(inliers, len(good_matches), inlier_ratio)

        return CandidateResult(
            artwork_id=reference.id,
            title=reference.title,
            good_matches=len(good_matches),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
        )


def compute_confidence(inliers: int, good_matches: int, inlier_ratio: float) -> float:
    inlier_score = min(1.0, inliers / 80.0)
    match_score = min(1.0, good_matches / 120.0)
    ratio_score = min(1.0, inlier_ratio / 0.55)

    return 0.55 * inlier_score + 0.20 * match_score + 0.25 * ratio_score


def decode_image_gray(image_bytes: bytes, max_dimension: int) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RecognitionError("Could not decode image. Upload a JPEG or PNG.")

    return to_gray_resized(image, max_dimension)


def load_image_gray(path: Path, max_dimension: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RecognitionError(f"Could not load image: {path}")

    return to_gray_resized(image, max_dimension)


def to_gray_resized(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)

    if largest > max_dimension:
        scale = max_dimension / largest

        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    return gray


def keypoint_to_array(kp) -> list[float]:
    return [
        kp.pt[0],
        kp.pt[1],
        kp.size,
        kp.angle,
        kp.response,
        kp.octave,
        kp.class_id,
    ]


def array_to_keypoint(row: np.ndarray):
    return cv2.KeyPoint(
        float(row[0]),
        float(row[1]),
        float(row[2]),
        float(row[3]),
        float(row[4]),
        int(row[5]),
        int(row[6]),
    )
