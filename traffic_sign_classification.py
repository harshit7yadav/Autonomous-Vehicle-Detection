import numpy as np
import cv2
from typing import List, Dict, Tuple

class TrafficSignClassifier:
    """Mock implementation of ResNet18-based traffic sign classifier"""
    
    def __init__(self):
        self.classes = [
            'stop', 'yield', 'speed_limit_30', 'speed_limit_50', 
            'speed_limit_70', 'no_entry', 'turn_left', 'turn_right',
            'straight_ahead', 'pedestrian_crossing'
        ]
        self.input_size = (224, 224)
        self.model_loaded = True
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect and classify traffic signs in the image
        
        Args:
            image: Input RGB image array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detection dictionaries with bbox, class, and confidence
        """
        detections = []
        
        # Convert to proper format for processing
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Use template matching and color filtering to find sign-like regions
        sign_regions = self._detect_sign_regions(image)
        
        for region in sign_regions:
            x, y, w, h = region
            
            # Extract region of interest
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
                
            # Mock classification - in reality this would be ResNet18 inference
            try:
                result = self._classify_region(roi)
                if result is None:
                    continue
                predicted_class, confidence = result
                
                # Only add detections with valid classes and sufficient confidence
                if confidence >= confidence_threshold and predicted_class != 'unknown':
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'class': predicted_class,
                        'confidence': confidence
                    })
            except Exception:
                # Skip this region if classification fails
                continue
        
        return detections
    
    def _detect_sign_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential traffic sign regions using improved computer vision"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        regions = []
        
        # Enhanced circular detection with stricter parameters
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80,
                                  param1=100, param2=50, minRadius=25, maxRadius=80)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Create bounding box around circle
                bbox_x = max(0, x-r)
                bbox_y = max(0, y-r)
                bbox_w = min(2*r, image.shape[1] - bbox_x)
                bbox_h = min(2*r, image.shape[0] - bbox_y)
                
                # Validate the circular region
                if self._validate_sign_region(image, hsv, bbox_x, bbox_y, bbox_w, bbox_h, 'circular'):
                    regions.append((bbox_x, bbox_y, bbox_w, bbox_h))
        
        # Enhanced rectangular detection with better filtering
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 4000:  # More restrictive size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # More restrictive aspect ratio for signs
                if 0.8 < aspect_ratio < 1.25 and w > 40 and h > 40:
                    # Validate the rectangular region
                    if self._validate_sign_region(image, hsv, x, y, w, h, 'rectangular'):
                        regions.append((x, y, w, h))
        
        return regions
    
    def _validate_sign_region(self, image: np.ndarray, hsv: np.ndarray, x: int, y: int, w: int, h: int, shape_type: str) -> bool:
        """Validate if a region is likely to be a traffic sign"""
        # Extract region
        roi = image[y:y+h, x:x+w]
        roi_hsv = hsv[y:y+h, x:x+w]
        
        if roi.size == 0 or roi_hsv.size == 0:
            return False
        
        # Check for sign-like colors (red, blue, yellow, white)
        red_mask = cv2.inRange(roi_hsv, (0, 50, 50), (10, 255, 255)) + \
                   cv2.inRange(roi_hsv, (170, 50, 50), (180, 255, 255))
        blue_mask = cv2.inRange(roi_hsv, (100, 50, 50), (130, 255, 255))
        yellow_mask = cv2.inRange(roi_hsv, (20, 50, 50), (30, 255, 255))
        white_mask = cv2.inRange(roi_hsv, (0, 0, 200), (180, 30, 255))
        
        # Calculate color ratios
        total_pixels = w * h
        red_ratio = np.sum(red_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        white_ratio = np.sum(white_mask > 0) / total_pixels
        
        # Must have significant sign-like colors
        sign_color_ratio = red_ratio + blue_ratio + yellow_ratio + white_ratio
        if sign_color_ratio < 0.2:  # At least 20% sign-like colors
            return False
        
        # Check position - traffic signs are typically in upper 2/3 of image
        if y > image.shape[0] * 0.75:  # Reject regions in bottom 25%
            return False
        
        # Check for uniform background (signs have distinct boundaries)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges_roi = cv2.Canny(gray_roi, 50, 150)
        edge_ratio = np.sum(edges_roi > 0) / total_pixels
        
        # Signs should have clear edges but not too many internal details
        if edge_ratio < 0.05 or edge_ratio > 0.4:
            return False
        
        # Additional validation for shape consistency
        if shape_type == 'circular':
            # For circular signs, check if the region is reasonably circular
            contours_roi, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_roi:
                largest_contour = max(contours_roi, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.4:  # Not circular enough
                        return False
        
        return True
    
    def _classify_region(self, roi: np.ndarray) -> Tuple[str, float]:
        """Enhanced classification of a region - simulates ResNet18 output with better accuracy"""
        # Safety check for empty ROI
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return 'unknown', 0.1
        
        # Resize to model input size
        try:
            roi_resized = cv2.resize(roi, self.input_size)
        except Exception:
            return 'unknown', 0.1
        
        # Enhanced feature extraction based on color, shape, and texture properties
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
        
        # More precise color detection
        red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) + \
                   cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 70, 50), (130, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 70, 50), (30, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        
        total_pixels = roi_resized.shape[0] * roi_resized.shape[1]
        red_ratio = np.sum(red_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        white_ratio = np.sum(white_mask > 0) / total_pixels
        
        # Shape analysis for better classification
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_circular = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                is_circular = circularity > 0.6
        
        # Enhanced classification logic with higher confidence thresholds
        base_confidence = 0.5
        
        if red_ratio > 0.25:
            if is_circular and red_ratio > 0.4:
                # Circular red signs (stop, no entry)
                if white_ratio > 0.15:  # Stop signs have white text/border
                    return 'stop', min(0.95, base_confidence + red_ratio + white_ratio)
                else:
                    return 'no_entry', min(0.88, base_confidence + red_ratio)
            elif red_ratio > 0.3:
                # Other red signs (yield, etc.)
                return 'yield', min(0.85, base_confidence + red_ratio)
        
        elif blue_ratio > 0.15 and white_ratio > 0.1:
            # Blue signs with white (typically mandatory/informational)
            if is_circular:
                return np.random.choice(['turn_left', 'turn_right', 'straight_ahead']), \
                       min(0.82, base_confidence + blue_ratio + white_ratio)
            else:
                # Rectangular blue signs (speed limits, etc.)
                return np.random.choice(['speed_limit_30', 'speed_limit_50', 'speed_limit_70']), \
                       min(0.85, base_confidence + blue_ratio + white_ratio)
        
        elif yellow_ratio > 0.2:
            # Warning signs (typically yellow)
            if white_ratio > 0.1:  # Yellow signs often have white symbols
                return 'pedestrian_crossing', min(0.78, base_confidence + yellow_ratio + white_ratio)
            else:
                return np.random.choice(['turn_left', 'turn_right']), \
                       min(0.70, base_confidence + yellow_ratio)
        
        else:
            # If no strong color patterns, likely not a traffic sign
            # Return low confidence to filter out false positives
            return 'unknown', np.random.uniform(0.2, 0.4)
    
    def get_class_names(self) -> List[str]:
        """Return list of supported traffic sign classes"""
        return self.classes.copy()
