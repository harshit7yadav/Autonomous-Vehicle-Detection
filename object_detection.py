import numpy as np
import cv2
from typing import List, Dict, Tuple

class ObjectDetectionModel:
    """Mock implementation of YOLOv8n object detection model"""
    
    def __init__(self):
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light']
        self.input_size = (640, 640)
        self.model_loaded = True
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in the image using mock YOLOv8n
        
        Args:
            image: Input RGB image array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detection dictionaries with bbox, class, and confidence
        """
        detections = []
        
        # Convert to proper format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Mock object detection using computer vision techniques
        # This simulates what YOLOv8n would output
        
        # Vehicle detection using edge detection and contours
        vehicle_detections = self._detect_vehicles(image, confidence_threshold)
        detections.extend(vehicle_detections)
        
        # Person detection using HOG-like features
        person_detections = self._detect_persons(image, confidence_threshold)
        detections.extend(person_detections)
        
        # Traffic light detection using color filtering
        traffic_light_detections = self._detect_traffic_lights(image, confidence_threshold)
        detections.extend(traffic_light_detections)
        
        return detections
    
    def _detect_vehicles(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced vehicle detection with improved accuracy"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Focus on lower 2/3 of image where vehicles typically appear
        roi_y = image.shape[0] // 3
        roi = gray[roi_y:, :]
        roi_color = image[roi_y:, :]
        
        # Enhanced edge detection with adaptive thresholding
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:  # More restrictive minimum size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adjust coordinates to full image
                y_full = y + roi_y
                
                # More strict aspect ratio filtering
                aspect_ratio = w / h
                if 1.3 < aspect_ratio < 2.8 and w > 60 and h > 40:
                    # Validate vehicle characteristics
                    roi_vehicle = roi_color[y:y+h, x:x+w]
                    if self._validate_vehicle_region(roi_vehicle, area):
                        # Determine vehicle type based on size and shape
                        vehicle_class, base_confidence = self._classify_vehicle(area, aspect_ratio, w, h)
                        
                        # Add position-based confidence boost (road level)
                        position_confidence = self._calculate_position_confidence(y_full, image.shape[0])
                        final_confidence = min(0.95, base_confidence * position_confidence)
                        
                        if final_confidence >= threshold:
                            detections.append({
                                'bbox': [x, y_full, x+w, y_full+h],
                                'class': vehicle_class,
                                'confidence': final_confidence
                            })
        
        return detections
    
    def _validate_vehicle_region(self, roi: np.ndarray, area: float) -> bool:
        """Validate if a region looks like a vehicle"""
        if roi.size == 0:
            return False
        
        # Check for vehicle-like colors (metal, paint colors)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Vehicle colors tend to be less saturated than bright objects
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        
        # Vehicles typically have moderate saturation (not too bright, not too dull)
        if avg_saturation > 180 or avg_saturation < 20:
            return False
        
        # Check for rectangular/geometric features
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        
        # Vehicles should have some edges but not too many (not people/trees)
        if edge_density < 0.02 or edge_density > 0.3:
            return False
        
        return True
    
    def _classify_vehicle(self, area: float, aspect_ratio: float, width: int, height: int) -> Tuple[str, float]:
        """Classify vehicle type based on size and shape"""
        if area > 12000 or width > 150:  # Large vehicles
            if aspect_ratio > 2.2:
                return 'bus', np.random.uniform(0.75, 0.90)
            else:
                return 'truck', np.random.uniform(0.70, 0.88)
        elif area > 4000:  # Medium vehicles
            if aspect_ratio > 2.0:
                return 'car', np.random.uniform(0.80, 0.95)
            else:
                return 'car', np.random.uniform(0.75, 0.90)
        else:  # Small vehicles
            if height < 60:  # Low profile
                return 'bicycle', np.random.uniform(0.65, 0.82)
            else:
                return 'motorcycle', np.random.uniform(0.60, 0.85)
    
    def _calculate_position_confidence(self, y_position: int, image_height: int) -> float:
        """Calculate confidence based on position (vehicles should be on ground level)"""
        # Vehicles should be in the lower portion of the image
        relative_position = y_position / image_height
        
        if relative_position > 0.4:  # Lower 60% of image
            return 1.0
        elif relative_position > 0.2:  # Middle area
            return 0.8
        else:  # Upper area (less likely for vehicles)
            return 0.6
    
    def _detect_persons(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced person detection with improved accuracy"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhanced preprocessing for person detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use adaptive thresholding for better contrast
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 2500:  # More restrictive person-sized range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Stricter person aspect ratio (taller than wide)
                aspect_ratio = h / w
                if 1.8 < aspect_ratio < 3.5 and h > 60 and w > 20:
                    # Validate person characteristics
                    roi = image[y:y+h, x:x+w]
                    if self._validate_person_region(roi, aspect_ratio):
                        # Calculate confidence based on multiple factors
                        base_confidence = self._calculate_person_confidence(x, y, w, h, image.shape, aspect_ratio)
                        
                        if base_confidence >= threshold:
                            detections.append({
                                'bbox': [x, y, x+w, y+h],
                                'class': 'person',
                                'confidence': base_confidence
                            })
        
        return detections
    
    def _validate_person_region(self, roi: np.ndarray, aspect_ratio: float) -> bool:
        """Validate if a region looks like a person"""
        if roi.size == 0:
            return False
        
        # Check for human-like color distribution
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # People typically have varied colors (clothing, skin)
        saturation = hsv[:, :, 1]
        saturation_variance = np.var(saturation)
        
        # Should have some color variation but not too extreme
        if saturation_variance < 100 or saturation_variance > 3000:
            return False
        
        # Check for vertical structure (people are vertical)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Check for head-like region (upper portion should be different from body)
        head_region = gray[:roi.shape[0]//3, :]
        body_region = gray[roi.shape[0]//3:, :]
        
        if head_region.size > 0 and body_region.size > 0:
            head_avg = np.mean(head_region)
            body_avg = np.mean(body_region)
            
            # Head and body should have some contrast
            if abs(head_avg - body_avg) < 10:
                return False
        
        return True
    
    def _calculate_person_confidence(self, x: int, y: int, w: int, h: int, image_shape: Tuple[int, int, int], aspect_ratio: float) -> float:
        """Calculate person detection confidence based on multiple factors"""
        base_confidence = 0.5
        
        # Position factor - people more likely on ground level or sidewalks
        relative_y = y / image_shape[0]
        if relative_y > 0.3:  # Lower 70% of image
            position_factor = 1.0
        else:
            position_factor = 0.7
        
        # Edge proximity factor - people often near sidewalks/edges
        edge_distance = min(x, image_shape[1] - (x + w))
        edge_factor = 1.0 if edge_distance < 150 else 0.8
        
        # Aspect ratio factor - closer to typical human proportions
        if 2.0 < aspect_ratio < 3.0:
            aspect_factor = 1.0
        else:
            aspect_factor = 0.8
        
        # Size factor - reasonable human size
        area = w * h
        if 500 < area < 1500:
            size_factor = 1.0
        else:
            size_factor = 0.9
        
        final_confidence = base_confidence * position_factor * edge_factor * aspect_factor * size_factor
        return min(0.85, max(0.3, final_confidence + np.random.uniform(-0.1, 0.1)))
    
    def _detect_traffic_lights(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced traffic light detection with improved accuracy"""
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # More precise color ranges for traffic lights
        # Red range (more restrictive)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        # Green range (more restrictive)
        green_lower = np.array([45, 100, 100])
        green_upper = np.array([75, 255, 255])
        
        # Yellow range (more restrictive)
        yellow_lower = np.array([22, 100, 100])
        yellow_upper = np.array([28, 255, 255])
        
        # Create individual masks
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Process each color separately for better accuracy
        for mask, color_name in [(red_mask, 'red'), (green_mask, 'green'), (yellow_mask, 'yellow')]:
            if np.sum(mask) == 0:  # Skip if no pixels of this color
                continue
                
            # Clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 800:  # Traffic light sized objects
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Validate traffic light characteristics
                    if self._validate_traffic_light_region(image, hsv, x, y, w, h, mask):
                        # Calculate confidence based on multiple factors
                        confidence = self._calculate_traffic_light_confidence(x, y, w, h, image.shape, mask, area)
                        
                        if confidence >= threshold:
                            detections.append({
                                'bbox': [x, y, x+w, y+h],
                                'class': 'traffic_light',
                                'confidence': confidence
                            })
        
        return detections
    
    def _validate_traffic_light_region(self, image: np.ndarray, hsv: np.ndarray, x: int, y: int, w: int, h: int, mask: np.ndarray) -> bool:
        """Validate if a region is likely a traffic light"""
        # Traffic lights should be in upper portion of image
        if y > image.shape[0] * 0.7:  # Reject if in bottom 30%
            return False
        
        # Traffic lights have specific aspect ratios
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.8 or aspect_ratio > 2.5:  # Not traffic light proportions
            return False
        
        # Check color intensity in the region
        roi_mask = mask[y:y+h, x:x+w]
        if roi_mask.size == 0:
            return False
            
        color_ratio = np.sum(roi_mask > 0) / (w * h)
        if color_ratio < 0.2:  # Insufficient color presence
            return False
        
        # Traffic lights should be bright
        roi_hsv = hsv[y:y+h, x:x+w]
        avg_value = np.mean(roi_hsv[:, :, 2])  # V channel (brightness)
        if avg_value < 100:  # Too dark to be an active traffic light
            return False
        
        return True
    
    def _calculate_traffic_light_confidence(self, x: int, y: int, w: int, h: int, image_shape: Tuple[int, int, int], mask: np.ndarray, area: float) -> float:
        """Calculate traffic light confidence based on multiple factors"""
        base_confidence = 0.6
        
        # Position factor - traffic lights in upper portion
        relative_y = y / image_shape[0]
        if relative_y < 0.3:  # Upper 30%
            position_factor = 1.0
        elif relative_y < 0.5:  # Middle area
            position_factor = 0.9
        else:
            position_factor = 0.7
        
        # Size factor - appropriate traffic light size
        if 100 < area < 500:
            size_factor = 1.0
        else:
            size_factor = 0.8
        
        # Color intensity factor
        roi_mask = mask[y:y+h, x:x+w]
        color_ratio = np.sum(roi_mask > 0) / (w * h) if (w * h) > 0 else 0
        intensity_factor = min(1.0, color_ratio * 2)  # Boost for strong color presence
        
        # Aspect ratio factor
        aspect_ratio = h / w if w > 0 else 0
        if 1.0 < aspect_ratio < 2.0:
            aspect_factor = 1.0
        else:
            aspect_factor = 0.9
        
        final_confidence = base_confidence * position_factor * size_factor * intensity_factor * aspect_factor
        return min(0.92, max(0.4, final_confidence + np.random.uniform(-0.05, 0.05)))
    
    def get_class_names(self) -> List[str]:
        """Return list of supported object classes"""
        return self.classes.copy()
