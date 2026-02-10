#!/usr/bin/env python3
"""
SAM3 Interface - Client for SAM3 Docker Server
Provides easy-to-use methods for object segmentation from ROS2 camera topics.
"""

import base64
import io
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image


@dataclass
class SegmentationMask:
    """Represents a single segmentation result."""
    id: int
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[int, int]  # cx, cy
    area: int
    mask: np.ndarray  # Binary mask (H, W)
    label: Optional[str] = None
    detection_confidence: Optional[float] = None


class SAM3Client:
    """
    Client for communicating with SAM3 Docker server.
    
    Usage:
        sam3 = SAM3Client()
        
        # Point-prompted segmentation
        masks = sam3.segment_point(image, points=[[320, 240]], labels=[1])
        
        # Detect all objects
        masks = sam3.detect_all(image)
        
        # Refine YOLO detections with SAM
        masks = sam3.segment_detections(image, detections)
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5005, timeout: int = 60):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if SAM3 server is available."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                print(f"[SAM3Client] Connected - Device: {info.get('device')}, CUDA: {info.get('cuda_available')}")
                return True
        except Exception as e:
            print(f"[SAM3Client] Warning: SAM3 server not available at {self.base_url}: {e}")
        return False
    
    def is_available(self) -> bool:
        """Check if server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert BGR to RGB if needed (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _decode_mask(self, mask_b64: str) -> np.ndarray:
        """Decode base64 mask to numpy array."""
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask = np.array(mask_img)
        return (mask > 127).astype(np.uint8)  # Binary mask
    
    def _parse_masks(self, response: dict) -> List[SegmentationMask]:
        """Parse API response into SegmentationMask objects."""
        masks = []
        for m in response.get('masks', []):
            masks.append(SegmentationMask(
                id=m['id'],
                score=m['score'],
                bbox=tuple(m['bbox']),
                centroid=tuple(m['centroid']),
                area=m['area'],
                mask=self._decode_mask(m['mask']),
                label=m.get('label'),
                detection_confidence=m.get('detection_confidence')
            ))
        return masks
    
    def segment_point(
        self, 
        image: np.ndarray, 
        points: List[Tuple[int, int]], 
        labels: Optional[List[int]] = None,
        multimask: bool = False
    ) -> List[SegmentationMask]:
        """
        Segment object at given point coordinates.
        
        Args:
            image: RGB/BGR image as numpy array
            points: List of (x, y) coordinates
            labels: 1 for foreground, 0 for background (default: all foreground)
            multimask: Return multiple mask proposals
            
        Returns:
            List of SegmentationMask objects
        """
        if labels is None:
            labels = [1] * len(points)
        
        payload = {
            "image": self._encode_image(image),
            "points": points,
            "labels": labels,
            "multimask": multimask
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/segment_point",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return self._parse_masks(resp.json())
        except Exception as e:
            print(f"[SAM3Client] segment_point error: {e}")
            return []
    
    def segment_box(
        self, 
        image: np.ndarray, 
        box: Tuple[int, int, int, int],
        multimask: bool = False
    ) -> List[SegmentationMask]:
        """
        Segment object within bounding box.
        
        Args:
            image: RGB/BGR image as numpy array
            box: (x1, y1, x2, y2) bounding box
            multimask: Return multiple mask proposals
            
        Returns:
            List of SegmentationMask objects
        """
        payload = {
            "image": self._encode_image(image),
            "box": list(box),
            "multimask": multimask
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/segment_box",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return self._parse_masks(resp.json())
        except Exception as e:
            print(f"[SAM3Client] segment_box error: {e}")
            return []
    
    def detect_all(
        self, 
        image: np.ndarray, 
        min_area: int = 100,
        max_masks: int = 50
    ) -> List[SegmentationMask]:
        """
        Automatic mask generation - detect all objects.
        
        Args:
            image: RGB/BGR image as numpy array
            min_area: Minimum mask area in pixels
            max_masks: Maximum number of masks to return
            
        Returns:
            List of SegmentationMask objects sorted by score
        """
        payload = {
            "image": self._encode_image(image),
            "min_area": min_area,
            "max_masks": max_masks
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/detect_all",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return self._parse_masks(resp.json())
        except Exception as e:
            print(f"[SAM3Client] detect_all error: {e}")
            return []

    def segment_text(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> List[SegmentationMask]:
        """
        Segment objects using a text prompt.
        
        Args:
            image: RGB/BGR image as numpy array
            prompt: Text prompt (e.g., "bottle", "red chair")
            
        Returns:
            List of SegmentationMask objects
        """
        payload = {
            "image": self._encode_image(image),
            "prompt": prompt
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/segment_text",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return self._parse_masks(resp.json())
        except Exception as e:
            print(f"[SAM3Client] segment_text error: {e}")
            return []
    
    def segment_detections(
        self, 
        image: np.ndarray, 
        detections: List[dict]
    ) -> List[SegmentationMask]:
        """
        Refine bounding box detections (e.g., from YOLO) with SAM.
        
        Args:
            image: RGB/BGR image as numpy array
            detections: List of dicts with keys:
                - box: [x1, y1, x2, y2]
                - label: str (optional)
                - confidence: float (optional)
                
        Returns:
            List of SegmentationMask objects with refined masks
        """
        payload = {
            "image": self._encode_image(image),
            "detections": detections
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/segment_with_detection",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return self._parse_masks(resp.json())
        except Exception as e:
            print(f"[SAM3Client] segment_detections error: {e}")
            return []


def visualize_masks(
    image: np.ndarray, 
    masks: List[SegmentationMask],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay masks on image for visualization.
    
    Args:
        image: Original image
        masks: List of SegmentationMask objects
        alpha: Transparency of mask overlay
        
    Returns:
        Image with colored mask overlays
    """
    output = image.copy()
    
    # Generate colors for each mask
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
    
    for i, seg_mask in enumerate(masks):
        color = colors[i].tolist()
        mask = seg_mask.mask
        
        # Create colored overlay
        overlay = output.copy()
        overlay[mask > 0] = color
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, seg_mask.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw centroid
        cx, cy = map(int, seg_mask.centroid)
        cv2.circle(output, (cx, cy), 5, color, -1)
        
        # Draw label
        label = seg_mask.label or f"obj_{seg_mask.id}"
        score_text = f"{label}: {seg_mask.score:.2f}"
        cv2.putText(output, score_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output
