#!/usr/bin/env python3
"""
ID Consistency Enforcer - Limits total unique player IDs to 4 throughout the video.
This wrapper around ByteTrack ensures that no more than 4 unique player IDs are ever created.
"""

import numpy as np
import supervision as sv
from typing import Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PlayerState:
    """Tracks the state of a player ID."""
    last_seen_frame: int
    total_detections: int
    last_position: Optional[tuple] = None
    confidence_history: list = None
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []

class IDConsistencyEnforcer:
    """
    Enforces that only 4 unique player IDs are used throughout the entire video.
    Acts as a wrapper around ByteTrack to prevent ID proliferation.
    """
    
    def __init__(self, max_players: int = 4, max_lost_frames: int = 90):
        """
        Args:
            max_players: Maximum number of unique player IDs allowed (4 for padel)
            max_lost_frames: Number of frames after which a lost player ID can be reused
        """
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.current_frame = 0
        
        # Track all player states
        self.player_states: Dict[int, PlayerState] = {}
        self.active_ids: Set[int] = set()
        self.available_ids: Set[int] = {1, 2, 3, 4}  # Pre-allocated IDs for padel
        self.next_id = 1
        
        # ID remapping for ByteTrack output
        self.bytetrack_to_consistent: Dict[int, int] = {}
        self.consistent_to_bytetrack: Dict[int, int] = {}
        
    def _calculate_position_similarity(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate normalized position similarity (0-1, higher = more similar)."""
        if pos1 is None or pos2 is None:
            return 0.0
        
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        # Normalize by image dimensions (assuming 1920x1080)
        normalized_distance = distance / np.sqrt(1920**2 + 1080**2)
        return max(0.0, 1.0 - normalized_distance * 10)  # Scale factor for sensitivity
    
    def _find_best_id_for_detection(self, detection_center: tuple, confidence: float) -> int:
        """
        Find the best existing ID for a new detection based on position and state.
        Returns an existing ID or creates a new one if needed.
        """
        # Check if any inactive IDs could be reactivated based on position
        best_match_id = None
        best_similarity = 0.5  # Minimum similarity threshold
        
        for player_id, state in self.player_states.items():
            if player_id not in self.active_ids:
                # Check if enough time has passed to consider reactivation
                frames_since_last_seen = self.current_frame - state.last_seen_frame
                if frames_since_last_seen <= self.max_lost_frames:
                    # Calculate position similarity
                    similarity = self._calculate_position_similarity(
                        detection_center, state.last_position
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = player_id
        
        if best_match_id is not None:
            return best_match_id
        
        # No good match found, try to assign a new ID if we haven't reached the limit
        if len(self.active_ids) < self.max_players and self.available_ids:
            return min(self.available_ids)
        
        # If we're at the limit, force reuse of the least recently seen ID
        if self.player_states:
            oldest_id = min(self.player_states.keys(), 
                          key=lambda x: self.player_states[x].last_seen_frame)
            return oldest_id
        
        # Fallback: return next available ID
        return self.next_id
    
    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Process detections and enforce ID consistency.
        
        Args:
            detections: ByteTrack detections with potentially inconsistent IDs
            
        Returns:
            Modified detections with consistent IDs (max 4 unique)
        """
        self.current_frame += 1
        
        if len(detections) == 0:
            return detections
        
        # Get detection centers for position-based matching
        detection_centers = []
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            detection_centers.append((center_x, center_y))
        
        # Create new consistent IDs for this frame
        new_tracker_ids = []
        self.active_ids.clear()
        
        for i, original_id in enumerate(detections.tracker_id):
            detection_center = detection_centers[i]
            confidence = detections.confidence[i] if detections.confidence is not None else 0.8
            
            # Check if we already have a mapping for this ByteTrack ID
            if original_id in self.bytetrack_to_consistent:
                consistent_id = self.bytetrack_to_consistent[original_id]
            else:
                # Find the best consistent ID for this detection
                consistent_id = self._find_best_id_for_detection(detection_center, confidence)
                
                # Update mappings
                self.bytetrack_to_consistent[original_id] = consistent_id
                self.consistent_to_bytetrack[consistent_id] = original_id
                
                # Remove from available IDs if it was available
                self.available_ids.discard(consistent_id)
            
            # Update player state
            if consistent_id not in self.player_states:
                self.player_states[consistent_id] = PlayerState(
                    last_seen_frame=self.current_frame,
                    total_detections=1,
                    last_position=detection_center,
                    confidence_history=[confidence]
                )
            else:
                state = self.player_states[consistent_id]
                state.last_seen_frame = self.current_frame
                state.total_detections += 1
                state.last_position = detection_center
                state.confidence_history.append(confidence)
                # Keep only recent confidence values
                if len(state.confidence_history) > 30:
                    state.confidence_history = state.confidence_history[-30:]
            
            # Add to active IDs
            self.active_ids.add(consistent_id)
            new_tracker_ids.append(consistent_id)
        
        # Clean up old mappings for IDs that haven't been seen for a long time
        current_bytetrack_ids = set(detections.tracker_id)
        for bt_id, cons_id in list(self.bytetrack_to_consistent.items()):
            if bt_id not in current_bytetrack_ids:
                frames_since_seen = self.current_frame - self.player_states[cons_id].last_seen_frame
                if frames_since_seen > self.max_lost_frames:
                    # Remove old mappings
                    del self.bytetrack_to_consistent[bt_id]
                    if cons_id in self.consistent_to_bytetrack:
                        del self.consistent_to_bytetrack[cons_id]
                    # Make ID available again
                    self.available_ids.add(cons_id)
        
        # Create new detections with consistent IDs
        new_detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.array(new_tracker_ids)
        )
        
        return new_detections
    
    def get_statistics(self) -> dict:
        """Get statistics about ID usage and consistency."""
        return {
            'total_unique_ids': len(self.player_states),
            'currently_active_ids': len(self.active_ids),
            'active_ids': sorted(self.active_ids),
            'max_players_limit': self.max_players,
            'current_frame': self.current_frame,
            'player_states': {
                pid: {
                    'total_detections': state.total_detections,
                    'last_seen_frame': state.last_seen_frame,
                    'frames_since_seen': self.current_frame - state.last_seen_frame
                }
                for pid, state in self.player_states.items()
            }
        }
