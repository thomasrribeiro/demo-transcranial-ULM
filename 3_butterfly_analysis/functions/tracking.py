"""
Bubble tracking functions for linking detections across frames.
Implements greedy tracking algorithm similar to MATLAB implementation.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Optional


def greedy_tracking(detections: List[np.ndarray],
                   max_distance: float = 10.0,
                   max_gap: int = 2,
                   min_track_length: int = 5) -> List[Dict]:
    """
    Greedy frame-to-frame bubble tracking with gap closing.

    Parameters
    ----------
    detections : list of np.ndarray
        List of detection arrays for each frame, each with shape (N, 2) containing
        (row, col) positions
    max_distance : float, default=10.0
        Maximum distance (pixels) for linking detections between frames
    max_gap : int, default=2
        Maximum number of frames to bridge gaps in tracks
    min_track_length : int, default=5
        Minimum track length to keep

    Returns
    -------
    list of dict
        List of tracks, each containing:
        - 'positions': (N, 2) array of positions
        - 'frames': array of frame indices
        - 'track_id': unique track identifier
        - 'length': track length
    """

    tracks = []
    active_tracks = []
    track_id_counter = 0

    for frame_idx, current_detections in enumerate(detections):
        if len(current_detections) == 0:
            # Update gap counter for active tracks
            for track in active_tracks:
                track['gap_count'] += 1
            continue

        # Match current detections to active tracks
        unmatched_detections = set(range(len(current_detections)))
        tracks_to_remove = []

        for track in active_tracks:
            # Get last position
            last_pos = track['positions'][-1]
            last_frame = track['frames'][-1]
            gap = frame_idx - last_frame

            if gap > max_gap:
                # Track has been lost
                tracks_to_remove.append(track)
                continue

            # Calculate distances to all current detections
            if len(current_detections) > 0:
                distances = cdist([last_pos], current_detections, 'euclidean')[0]

                # Find closest unmatched detection within max_distance
                valid_matches = []
                for det_idx in unmatched_detections:
                    if distances[det_idx] <= max_distance * (gap + 1):
                        valid_matches.append((det_idx, distances[det_idx]))

                if valid_matches:
                    # Take closest match
                    best_match_idx = min(valid_matches, key=lambda x: x[1])[0]

                    # Update track
                    track['positions'].append(current_detections[best_match_idx])
                    track['frames'].append(frame_idx)
                    track['gap_count'] = 0
                    track['length'] += 1

                    # Remove from unmatched
                    unmatched_detections.remove(best_match_idx)
                else:
                    # No match found, increment gap counter
                    track['gap_count'] += 1

        # Remove lost tracks
        tracks_to_remove_ids = set()
        for track in tracks_to_remove:
            if track['length'] >= min_track_length:
                tracks.append({
                    'positions': np.array(track['positions']),
                    'frames': np.array(track['frames']),
                    'track_id': track['track_id'],
                    'length': track['length']
                })
            tracks_to_remove_ids.add(track['track_id'])

        # Remove tracks from active list using track IDs
        active_tracks = [t for t in active_tracks if t['track_id'] not in tracks_to_remove_ids]

        # Check for tracks that exceeded max gap
        tracks_to_remove = []
        for track in active_tracks:
            if track['gap_count'] > max_gap:
                tracks_to_remove.append(track)

        tracks_to_remove_ids = set()
        for track in tracks_to_remove:
            if track['length'] >= min_track_length:
                tracks.append({
                    'positions': np.array(track['positions']),
                    'frames': np.array(track['frames']),
                    'track_id': track['track_id'],
                    'length': track['length']
                })
            tracks_to_remove_ids.add(track['track_id'])

        # Remove tracks from active list using track IDs
        active_tracks = [t for t in active_tracks if t['track_id'] not in tracks_to_remove_ids]

        # Start new tracks for unmatched detections
        for det_idx in unmatched_detections:
            active_tracks.append({
                'positions': [current_detections[det_idx]],
                'frames': [frame_idx],
                'track_id': track_id_counter,
                'gap_count': 0,
                'length': 1
            })
            track_id_counter += 1

    # Add remaining active tracks
    for track in active_tracks:
        if track['length'] >= min_track_length:
            tracks.append({
                'positions': np.array(track['positions']),
                'frames': np.array(track['frames']),
                'track_id': track['track_id'],
                'length': track['length']
            })

    return tracks


def track_bubbles(detections: List[Tuple[np.ndarray, np.ndarray]],
                 use_subpixel: bool = True,
                 **tracking_params) -> np.ndarray:
    """
    Track bubbles and format output similar to MATLAB Bubbles_array.

    Parameters
    ----------
    detections : list of tuple
        List of (pixel_positions, subpixel_positions) for each frame
    use_subpixel : bool, default=True
        Whether to use sub-pixel positions
    **tracking_params : dict
        Parameters for greedy_tracking

    Returns
    -------
    np.ndarray
        Array with columns: [x, z, vx, vz, track_length, original_x, original_z, frame_num]
        Similar to MATLAB Bubbles_positions_and_speed_table
    """

    # Extract positions for tracking
    if use_subpixel:
        positions = [det[1] if len(det[1]) > 0 else np.array([])
                    for det in detections]
    else:
        positions = [det[0] if len(det[0]) > 0 else np.array([])
                    for det in detections]

    # Perform tracking
    tracks = greedy_tracking(positions, **tracking_params)

    if not tracks:
        return np.array([])

    # Format output array
    bubble_array = []

    for track in tracks:
        positions = track['positions']
        frames = track['frames']
        track_length = track['length']

        # Calculate velocities (pixel/frame)
        if track_length > 1:
            dx = np.diff(positions[:, 1])  # x is column
            dz = np.diff(positions[:, 0])  # z is row
            dt = np.diff(frames)

            vx = dx / dt
            vz = dz / dt

            # Pad velocities (repeat last value)
            vx = np.append(vx, vx[-1] if len(vx) > 0 else 0)
            vz = np.append(vz, vz[-1] if len(vz) > 0 else 0)
        else:
            vx = np.zeros(track_length)
            vz = np.zeros(track_length)

        # Add each position in track to output array
        for i in range(track_length):
            bubble_array.append([
                positions[i, 1],    # x (column)
                positions[i, 0],    # z (row)
                vx[i],              # vx
                vz[i],              # vz
                track_length,       # track length
                int(positions[i, 1]),  # original pixel x
                int(positions[i, 0]),  # original pixel z
                frames[i]           # frame number
            ])

    return np.array(bubble_array)


def link_trajectories(tracks: List[Dict],
                     max_distance: float = 20.0,
                     max_time_gap: int = 5) -> List[Dict]:
    """
    Link broken trajectories that might belong to the same bubble.

    Parameters
    ----------
    tracks : list of dict
        List of tracks from greedy_tracking
    max_distance : float, default=20.0
        Maximum distance for linking track ends
    max_time_gap : int, default=5
        Maximum time gap between tracks to link

    Returns
    -------
    list of dict
        Linked tracks
    """

    if len(tracks) <= 1:
        return tracks

    linked_tracks = []
    used_tracks = set()

    for i, track1 in enumerate(tracks):
        if i in used_tracks:
            continue

        linked_track = {
            'positions': list(track1['positions']),
            'frames': list(track1['frames']),
            'track_id': track1['track_id'],
            'length': track1['length']
        }

        # Try to link with other tracks
        last_frame = linked_track['frames'][-1]
        last_pos = linked_track['positions'][-1]

        for j, track2 in enumerate(tracks):
            if j <= i or j in used_tracks:
                continue

            first_frame = track2['frames'][0]
            first_pos = track2['positions'][0]

            # Check time gap
            time_gap = first_frame - last_frame
            if 0 < time_gap <= max_time_gap:
                # Check distance
                distance = np.linalg.norm(first_pos - last_pos)
                if distance <= max_distance:
                    # Link tracks
                    linked_track['positions'].extend(track2['positions'])
                    linked_track['frames'].extend(track2['frames'])
                    linked_track['length'] += track2['length']

                    used_tracks.add(j)

                    # Update for next potential link
                    last_frame = linked_track['frames'][-1]
                    last_pos = linked_track['positions'][-1]

        # Convert back to arrays
        linked_track['positions'] = np.array(linked_track['positions'])
        linked_track['frames'] = np.array(linked_track['frames'])

        linked_tracks.append(linked_track)
        used_tracks.add(i)

    return linked_tracks


def calculate_track_velocities(track: Dict,
                              pixel_size: float = 0.1,
                              framerate: float = 800.0) -> Dict:
    """
    Calculate velocities and other track metrics.

    Parameters
    ----------
    track : dict
        Track dictionary from tracking
    pixel_size : float, default=0.1
        Pixel size in mm
    framerate : float, default=800.0
        Frame rate in Hz

    Returns
    -------
    dict
        Track with added velocity metrics
    """

    positions = track['positions']
    frames = track['frames']

    if len(positions) > 1:
        # Calculate displacements
        dx = np.diff(positions[:, 0]) * pixel_size  # mm
        dy = np.diff(positions[:, 1]) * pixel_size  # mm
        dt = np.diff(frames) / framerate  # seconds

        # Velocities
        vx = dx / dt  # mm/s
        vy = dy / dt  # mm/s
        speed = np.sqrt(vx**2 + vy**2)

        track['velocities'] = {
            'vx': vx,
            'vy': vy,
            'speed': speed,
            'mean_speed': np.mean(speed),
            'std_speed': np.std(speed)
        }
    else:
        track['velocities'] = {
            'vx': np.array([]),
            'vy': np.array([]),
            'speed': np.array([]),
            'mean_speed': 0,
            'std_speed': 0
        }

    return track