"""
Enhanced Player Tracker with Identity Preservation

This module addresses the player ID switching issue by implementing:
1. Maximum 4 players limit
2. Improved ByteTrack configuration
3. Appearance-based re-identification using clothing colors
4. Identity consistency checks
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import supervision as sv
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine


class PlayerAppearanceFeatures:
    """Extract and compare player appearance features for re-identification."""
    
    def __init__(self):
        self.color_cache = {}
        self.position_history = defaultdict(lambda: deque(maxlen=30))  # 1 second at 30fps
    
    def extract_dominant_colors(self, frame: np.ndarray, bbox: List[float], n_colors: int = 3) -> np.ndarray:
        """Extract dominant clothing colors from player bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract player region with some padding to focus on clothing
        h, w = y2 - y1, x2 - x1
        y1_crop = max(0, y1 + int(h * 0.2))  # Skip head region
        y2_crop = min(frame.shape[0], y2 - int(h * 0.1))  # Skip feet region
        x1_crop = max(0, x1 + int(w * 0.1))  # Small horizontal padding
        x2_crop = min(frame.shape[1], x2 - int(w * 0.1))
        
        if y2_crop <= y1_crop or x2_crop <= x1_crop:
            return np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]])
        
        player_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # Convert to HSV for better color representation
        hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        
        # Reshape for clustering
        pixels = hsv_region.reshape(-1, 3)
        
        # Remove very dark/bright pixels (likely shadows/highlights)
        valid_pixels = pixels[(pixels[:, 2] > 30) & (pixels[:, 2] < 220)]
        
        if len(valid_pixels) < 10:
            return np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]])
        
        # K-means clustering to find dominant colors
        try:
            kmeans = KMeans(n_clusters=min(n_colors, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            # Convert back to BGR for consistency
            dominant_colors_bgr = []
            for color in dominant_colors:
                hsv_color = np.uint8([[color]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                dominant_colors_bgr.append(bgr_color)
            
            return np.array(dominant_colors_bgr)
        except:
            return np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]])
    
    def calculate_color_similarity(self, colors1: np.ndarray, colors2: np.ndarray) -> float:
        """Calculate similarity between two sets of dominant colors."""
        if len(colors1) == 0 or len(colors2) == 0:
            return 0.0
        
        # Normalize colors
        colors1_norm = colors1.astype(float) / 255.0
        colors2_norm = colors2.astype(float) / 255.0
        
        # Calculate pairwise similarities and take the best matches
        similarities = []
        for c1 in colors1_norm:
            max_sim = 0
            for c2 in colors2_norm:
                sim = 1.0 - cosine(c1, c2)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)
        
        return np.mean(similarities)
    
    def calculate_position_similarity(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate position similarity based on distance."""
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        # Normalize by image diagonal (assuming 1920x1080)
        max_distance = np.sqrt(1920**2 + 1080**2)
        return max(0, 1.0 - distance / (max_distance * 0.3))  # 30% of diagonal as max reasonable distance


class EnhancedPlayerTracker:
    """Enhanced player tracker with identity preservation."""
    
    def __init__(self, original_tracker):
        self.original_tracker = original_tracker
        self.appearance_features = PlayerAppearanceFeatures()
        self.player_profiles = {}  # id -> {colors, positions, confidence}
        self.id_mapping = {}  # old_id -> consistent_id
        self.next_consistent_id = 1
        self.max_players = 4
        
        # Enhanced ByteTrack configuration
        self.setup_enhanced_tracking()
    
    def setup_enhanced_tracking(self):
        """Configure ByteTrack for better identity preservation."""
        if hasattr(self.original_tracker, 'byte_track'):
            # More conservative tracking parameters
            self.original_tracker.byte_track.track_thresh = 0.6  # Higher confidence threshold
            self.original_tracker.byte_track.track_buffer = 60  # 2 seconds buffer at 30fps
            self.original_tracker.byte_track.match_thresh = 0.8  # Stricter matching
            self.original_tracker.byte_track.frame_rate = 30
    
    def create_player_profile(self, player_id: int, frame: np.ndarray, bbox: List[float], position: Tuple[float, float]):
        """Create appearance profile for a new player."""
        colors = self.appearance_features.extract_dominant_colors(frame, bbox)
        
        self.player_profiles[player_id] = {
            'colors': colors,
            'positions': deque([position], maxlen=30),
            'confidence': 1.0,
            'frames_seen': 1
        }
    
    def update_player_profile(self, player_id: int, frame: np.ndarray, bbox: List[float], position: Tuple[float, float]):
        """Update existing player profile."""
        if player_id in self.player_profiles:
            profile = self.player_profiles[player_id]
            profile['positions'].append(position)
            profile['frames_seen'] += 1
            
            # Periodically update color profile
            if profile['frames_seen'] % 30 == 0:  # Every 1 second
                new_colors = self.appearance_features.extract_dominant_colors(frame, bbox)
                # Blend with existing colors
                profile['colors'] = (profile['colors'] * 0.7 + new_colors * 0.3).astype(int)
    
    def find_best_match(self, new_player_id: int, frame: np.ndarray, bbox: List[float], position: Tuple[float, float]) -> Optional[int]:
        """Find the best matching existing player for re-identification."""
        if len(self.player_profiles) == 0:
            return None
        
        new_colors = self.appearance_features.extract_dominant_colors(frame, bbox)
        best_match_id = None
        best_score = 0.0
        
        for existing_id, profile in self.player_profiles.items():
            # Skip if this profile is already mapped to an active player
            if any(mapped_id == existing_id for mapped_id in self.id_mapping.values()):
                continue
            
            # Calculate appearance similarity
            color_sim = self.appearance_features.calculate_color_similarity(new_colors, profile['colors'])
            
            # Calculate position similarity (compare with recent positions)
            recent_positions = list(profile['positions'])[-10:]  # Last 10 positions
            if recent_positions:
                avg_position = np.mean(recent_positions, axis=0)
                pos_sim = self.appearance_features.calculate_position_similarity(position, tuple(avg_position))
            else:
                pos_sim = 0.0
            
            # Combined similarity score
            combined_score = color_sim * 0.7 + pos_sim * 0.3
            
            if combined_score > best_score and combined_score > 0.6:  # Minimum threshold
                best_score = combined_score
                best_match_id = existing_id
        
        return best_match_id
    
    def get_consistent_id(self, detected_id: int, frame: np.ndarray, bbox: List[float]) -> int:
        """Get consistent player ID, handling re-identification."""
        position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # If this detected_id is already mapped, use the consistent mapping
        if detected_id in self.id_mapping:
            consistent_id = self.id_mapping[detected_id]
            self.update_player_profile(consistent_id, frame, bbox, position)
            return consistent_id
        
        # Try to find a match with existing profiles
        best_match = self.find_best_match(detected_id, frame, bbox, position)
        
        if best_match is not None:
            # Re-identification: map to existing player
            self.id_mapping[detected_id] = best_match
            self.update_player_profile(best_match, frame, bbox, position)
            print(f"Re-identified player: detected_id {detected_id} -> consistent_id {best_match}")
            return best_match
        else:
            # New player: create new consistent ID if under limit
            if len(self.player_profiles) < self.max_players:
                consistent_id = self.next_consistent_id
                self.next_consistent_id += 1
                self.id_mapping[detected_id] = consistent_id
                self.create_player_profile(consistent_id, frame, bbox, position)
                print(f"New player: detected_id {detected_id} -> consistent_id {consistent_id}")
                return consistent_id
            else:
                # If at max players, try to reuse the least recently seen ID
                least_recent_id = min(self.player_profiles.keys(), 
                                    key=lambda x: self.player_profiles[x]['frames_seen'])
                self.id_mapping[detected_id] = least_recent_id
                self.update_player_profile(least_recent_id, frame, bbox, position)
                print(f"Reusing player ID: detected_id {detected_id} -> consistent_id {least_recent_id}")
                return least_recent_id
    
    def process_predictions(self, predictions: List, frame: np.ndarray) -> List:
        """Process predictions and apply identity consistency."""
        if not predictions:
            return predictions
        
        # Clean up old mappings for IDs not seen recently
        active_detected_ids = set()
        for player_obj in predictions[0].players if predictions else []:
            if player_obj.id is not None:
                active_detected_ids.add(player_obj.id)
        
        # Remove mappings for IDs not seen in recent frames
        self.id_mapping = {k: v for k, v in self.id_mapping.items() if k in active_detected_ids}
        
        # Apply consistent IDs
        for player_obj in predictions[0].players if predictions else []:
            if player_obj.id is not None:
                consistent_id = self.get_consistent_id(player_obj.id, frame, player_obj.xyxy)
                player_obj.id = consistent_id
        
        return predictions


def create_enhanced_player_tracker(original_tracker):
    """Factory function to create enhanced player tracker."""
    return EnhancedPlayerTracker(original_tracker)
