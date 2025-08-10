#!/usr/bin/env python3
"""
Script to analyze player tracking data and identify ID switching issues.
"""

import json
import sys
from collections import defaultdict

def analyze_player_tracking():
    """Analyze player detection data for ID consistency issues."""
    
    try:
        # Load player detections
        with open('cache/players_detections.json', 'r') as f:
            data = json.load(f)
        
        print(f"Total frames analyzed: {len(data)}")
        print(f"Video FPS: 30 (estimated)")
        
        # Track player IDs over time
        player_id_stats = defaultdict(list)
        frames_with_more_than_4_players = []
        
        for frame_idx, frame_data in enumerate(data):
            player_ids = [player['id'] for player in frame_data]
            unique_ids = set(player_ids)
            
            # Record each player ID appearance
            for pid in unique_ids:
                player_id_stats[pid].append(frame_idx)
            
            # Flag frames with more than 4 players
            if len(unique_ids) > 4:
                frames_with_more_than_4_players.append({
                    'frame': frame_idx,
                    'time': frame_idx / 30.0,
                    'player_ids': sorted(unique_ids),
                    'count': len(unique_ids)
                })
        
        print(f"\n=== PLAYER ID STATISTICS ===")
        print(f"Unique player IDs found: {sorted(player_id_stats.keys())}")
        
        for pid in sorted(player_id_stats.keys()):
            frames = player_id_stats[pid]
            first_frame = min(frames)
            last_frame = max(frames)
            print(f"Player ID {pid}: appears in {len(frames)} frames, "
                  f"from frame {first_frame} ({first_frame/30:.2f}s) to frame {last_frame} ({last_frame/30:.2f}s)")
        
        print(f"\n=== FRAMES WITH >4 PLAYERS ===")
        if frames_with_more_than_4_players:
            for issue in frames_with_more_than_4_players:
                print(f"Frame {issue['frame']} (t={issue['time']:.2f}s): "
                      f"{issue['count']} players with IDs {issue['player_ids']}")
        else:
            print("No frames found with more than 4 players")
        
        # Focus on the specific time mentioned (second 14)
        target_time = 14.0
        target_frame = int(target_time * 30)
        
        print(f"\n=== ANALYSIS AROUND SECOND 14 (frame {target_frame}) ===")
        start_frame = max(0, target_frame - 30)  # 1 second before
        end_frame = min(len(data), target_frame + 30)  # 1 second after
        
        for frame_idx in range(start_frame, end_frame, 5):  # Every 5 frames
            if frame_idx < len(data):
                frame_data = data[frame_idx]
                player_ids = sorted([player['id'] for player in frame_data])
                print(f"Frame {frame_idx} (t={frame_idx/30:.2f}s): Player IDs = {player_ids}")
        
        # Look for ID switches around that time
        print(f"\n=== ID SWITCH DETECTION ===")
        for frame_idx in range(1, len(data)):
            prev_frame = data[frame_idx - 1]
            curr_frame = data[frame_idx]
            
            prev_ids = set(player['id'] for player in prev_frame)
            curr_ids = set(player['id'] for player in curr_frame)
            
            # Check for new IDs appearing
            new_ids = curr_ids - prev_ids
            lost_ids = prev_ids - curr_ids
            
            if new_ids and frame_idx / 30.0 >= 13.0 and frame_idx / 30.0 <= 15.0:
                print(f"Frame {frame_idx} (t={frame_idx/30:.2f}s): "
                      f"New IDs: {sorted(new_ids)}, Lost IDs: {sorted(lost_ids)}")
        
    except FileNotFoundError:
        print("Error: cache/players_detections.json not found")
        return False
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    analyze_player_tracking()
