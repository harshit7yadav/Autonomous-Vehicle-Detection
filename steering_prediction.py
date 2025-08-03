import numpy as np
import cv2

class SteeringPredictionModel:
    """Mock implementation of NVIDIA PilotNet-inspired steering prediction model"""
    
    def __init__(self):
        self.input_size = (200, 66)  # PilotNet input dimensions
        self.model_loaded = True
        self.max_steering_angle = 45.0  # Maximum steering angle in degrees
    
    def predict(self, image: np.ndarray) -> float:
        """
        Predict steering angle from input image
        
        Args:
            image: Input RGB image array
            
        Returns:
            Predicted steering angle in degrees (-45 to +45)
        """
        # Preprocess image for steering prediction
        processed_image = self._preprocess_for_steering(image)
        
        # Mock steering prediction using computer vision techniques
        # This simulates what a trained PilotNet would output
        steering_angle = self._calculate_steering_from_lanes(processed_image)
        
        # Clamp to valid range
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        return float(steering_angle)
    
    def _preprocess_for_steering(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image following PilotNet methodology"""
        # Convert to proper format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Crop to focus on road area (remove sky and hood)
        height, width = image.shape[:2]
        crop_top = int(height * 0.35)  # Remove top 35% (sky)
        crop_bottom = int(height * 0.85)  # Remove bottom 15% (hood)
        
        cropped = image[crop_top:crop_bottom, :]
        
        # Resize to PilotNet input size
        resized = cv2.resize(cropped, self.input_size)
        
        # Convert to YUV color space (as used in PilotNet)
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Normalize pixel values
        normalized = yuv.astype(np.float32) / 255.0
        
        return normalized
    
    def _calculate_steering_from_lanes(self, image: np.ndarray) -> float:
        """Calculate steering angle based on lane detection"""
        # Convert back to uint8 for OpenCV operations
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Convert YUV to RGB first, then to grayscale
        if len(image_uint8.shape) == 3:
            rgb_image = cv2.cvtColor(image_uint8, cv2.COLOR_YUV2RGB)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_uint8
        
        # Edge detection for lane markings
        edges = cv2.Canny(gray, 50, 150)
        
        # Define region of interest (trapezoid focusing on lanes ahead)
        height, width = edges.shape
        roi_vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.4), int(height * 0.6)),
            (int(width * 0.6), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], dtype=np.int32)
        
        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_vertices[0]], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough line detection
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30,
                               minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return 0.0  # No lanes detected, go straight
        
        # Separate left and right lane lines
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                
                # Classify as left or right lane based on slope and position
                if slope < -0.3 and x1 < width // 2:  # Left lane (negative slope, left side)
                    left_lines.append(line[0])
                elif slope > 0.3 and x1 > width // 2:  # Right lane (positive slope, right side)
                    right_lines.append(line[0])
        
        # Calculate lane center
        lane_center_x = self._calculate_lane_center(left_lines, right_lines, width, height)
        
        # Calculate steering angle based on deviation from center
        image_center_x = width // 2
        deviation = lane_center_x - image_center_x
        
        # Convert pixel deviation to steering angle
        # This is a simplified model - real PilotNet learns this mapping
        steering_angle = (deviation / image_center_x) * self.max_steering_angle
        
        # Add some noise to simulate real model uncertainty
        noise = np.random.normal(0, 2.0)  # Small amount of noise
        steering_angle += noise
        
        return steering_angle
    
    def _calculate_lane_center(self, left_lines: list, right_lines: list, width: int, height: int) -> float:
        """Calculate the center point between left and right lanes"""
        
        def get_lane_line_x(lines, y_pos):
            """Get x coordinate of lane line at given y position"""
            if not lines:
                return None
            
            # Average all line segments to get a representative line
            x_coords = []
            for x1, y1, x2, y2 in lines:
                if y2 != y1:  # Avoid division by zero
                    # Linear interpolation to find x at y_pos
                    x = x1 + (x2 - x1) * (y_pos - y1) / (y2 - y1)
                    x_coords.append(x)
            
            return np.mean(x_coords) if x_coords else None
        
        # Calculate lane positions at the bottom of the image
        y_eval = height - 1
        
        left_x = get_lane_line_x(left_lines, y_eval)
        right_x = get_lane_line_x(right_lines, y_eval)
        
        # Calculate lane center
        if left_x is not None and right_x is not None:
            return float((left_x + right_x) / 2)
        elif left_x is not None:
            # Only left lane detected, estimate right lane
            estimated_right_x = left_x + width * 0.4  # Assume standard lane width
            return float((left_x + estimated_right_x) / 2)
        elif right_x is not None:
            # Only right lane detected, estimate left lane
            estimated_left_x = right_x - width * 0.4  # Assume standard lane width
            return float((estimated_left_x + right_x) / 2)
        else:
            # No lanes detected, assume center
            return float(width / 2)
    
    def get_steering_range(self) -> tuple:
        """Return the range of possible steering angles"""
        return (-self.max_steering_angle, self.max_steering_angle)
