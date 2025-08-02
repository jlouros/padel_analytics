#!/usr/bin/env python3
"""
Enhanced script to analyze player ID consistency and track creation patterns.
"""

import json
import sys
from collections import defaultdict, OrderedDict

def analyze_id_consistency():
    """Detailed analysis of player ID consistency and creation patterns."""
    
    try:
        # Load player detections
        with open('cache/players_detections.json', 'r') as f:
            data = json.load(f)
        
        print(f"🎬 Total frames analyzed: {len(data)}")
        print(f"📹 Video FPS: 30 (estimated)")
        print(f"⏱️  Video duration: {len(data)/30:.1f} seconds")
        
        # Track detailed player statistics
        player_id_stats = defaultdict(list)
        id_creation_timeline = []
        active_players_per_frame = []
        
        for frame_idx, frame_data in enumerate(data):
            player_ids = [player['id'] for player in frame_data]
            unique_ids = set(player_ids)
            
            # Record active players count
            active_players_per_frame.append(len(unique_ids))
            
            # Track each player ID appearance
            for pid in unique_ids:
                player_id_stats[pid].append(frame_idx)
                
                # Check if this is a new ID
                if len(player_id_stats[pid]) == 1:  # First appearance
                    id_creation_timeline.append({
                        'id': pid,
                        'frame': frame_idx,
                        'time': frame_idx / 30.0
                    })
        
        print(f"\n🆔 === PLAYER ID OVERVIEW ===")
        print(f"Total unique player IDs: {len(player_id_stats)}")
        print(f"Expected IDs: 4 (for padel)")
        print(f"Excess IDs created: {len(player_id_stats) - 4}")
        
        print(f"\n📊 === ID CREATION TIMELINE ===")
        for creation in id_creation_timeline:
            duration_frames = len(player_id_stats[creation['id']])
            duration_seconds = duration_frames / 30.0
            print(f"Player ID {creation['id']:2d}: Created at frame {creation['frame']:4d} "
                  f"({creation['time']:6.2f}s), active for {duration_seconds:6.1f}s ({duration_frames:4d} frames)")
        
        # Analyze active player count distribution
        print(f"\n👥 === ACTIVE PLAYERS PER FRAME DISTRIBUTION ===")
        from collections import Counter
        player_count_dist = Counter(active_players_per_frame)
        for count in sorted(player_count_dist.keys()):
            frames = player_count_dist[count]
            percentage = frames / len(data) * 100
            print(f"{count} players: {frames:4d} frames ({percentage:5.1f}%)")
        
        # Find problematic periods with many active IDs
        print(f"\n⚠️  === PROBLEMATIC PERIODS ===")
        time_windows = []
        window_size = 300  # 10 seconds at 30fps
        
        for start_frame in range(0, len(data), window_size):
            end_frame = min(start_frame + window_size, len(data))
            window_data = data[start_frame:end_frame]
            
            unique_ids_in_window = set()
            for frame_data in window_data:
                for player in frame_data:
                    unique_ids_in_window.add(player['id'])
            
            if len(unique_ids_in_window) > 4:
                time_windows.append({
                    'start_time': start_frame / 30.0,
                    'end_time': end_frame / 30.0,
                    'unique_ids': len(unique_ids_in_window),
                    'ids': sorted(unique_ids_in_window)
                })
        
        if time_windows:
            for window in time_windows:
                print(f"Time {window['start_time']:6.1f}s - {window['end_time']:6.1f}s: "
                      f"{window['unique_ids']} unique IDs {window['ids']}")
        else:
            print("No 10-second windows with >4 unique IDs found")
        
        # Analyze ID gaps and switches
        print(f"\n🔄 === ID CONTINUITY ANALYSIS ===")
        long_lived_ids = []
        short_lived_ids = []
        
        for pid, frames in player_id_stats.items():
            duration = len(frames)
            if duration > 300:  # >10 seconds
                long_lived_ids.append((pid, duration))
            elif duration < 60:  # <2 seconds
                short_lived_ids.append((pid, duration))
        
        print(f"Long-lived IDs (>10s): {len(long_lived_ids)}")
        for pid, duration in sorted(long_lived_ids, key=lambda x: x[1], reverse=True):
            print(f"  Player ID {pid:2d}: {duration:4d} frames ({duration/30:6.1f}s)")
        
        print(f"\nShort-lived IDs (<2s): {len(short_lived_ids)}")
        for pid, duration in sorted(short_lived_ids, key=lambda x: x[1]):
            print(f"  Player ID {pid:2d}: {duration:4d} frames ({duration/30:6.1f}s)")
        
        # Analysis summary
        print(f"\n📋 === SUMMARY ===")
        ideal_scenario = "4 players tracked consistently throughout video"
        current_reality = f"{len(player_id_stats)} unique IDs with frequent ID switches"
        print(f"Ideal: {ideal_scenario}")
        print(f"Reality: {current_reality}")
        print(f"Root cause: ByteTrack creating new IDs when re-detecting lost players")
        
        # Recommendations
        print(f"\n💡 === RECOMMENDATIONS ===")
        print("1. 🔧 Implement ID consistency enforcement")
        print("2. 🎨 Deploy appearance-based re-identification")
        print("3. 📏 Use positional constraints for player association")
        print("4. 🧠 Add memory buffer to prevent ID loss")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: cache/players_detections.json not found")
        return False
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False

if __name__ == "__main__":
    analyze_id_consistency()
