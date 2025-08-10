# 3D Court Model and 14-Keypoint Selection System

## Overview

The Padel Analytics system supports an advanced 3D court model that uses 14 keypoints to create a more accurate spatial representation of the padel court. This system enables enhanced ball trajectory tracking, 3D positioning, and improved analytics compared to the standard 12-keypoint system.

## Status

**Current Status**: **DISABLED** for compatibility reasons

- The system currently uses a 12-keypoint implementation for stability
- The 3D court model has been temporarily disabled in `main.py` (line 110-111)
- All core functionality works without the 3D model

## 12-Keypoint vs 14-Keypoint Systems

### 12-Keypoint System (Currently Active)

- Uses ground-level court boundary points only
- Provides 2D court mapping and homography
- Compatible with all current trackers
- Sufficient for player tracking and basic ball analysis

### 14-Keypoint System (Advanced, Currently Disabled)

- Includes 12 ground-level points + 2 height reference points
- Enables full 3D court modeling
- Provides enhanced ball trajectory analysis
- Better spatial accuracy for advanced analytics

## 14-Keypoint Selection Guide

### Keypoint Layout

```none
Ground Level (12 points):              Height Level (2 additional points):
k11--------------------k12             
|                       |              k13 ---- k14 (on back wall)
k8-----------k9--------k10             |          |
|            |          |              |          | (wall height)
k6----------------------k7             |          |
|            |          |              |          |
k3-----------k4---------k5             |          |
|                       |              |          |
k1----------------------k2             |          |
                                      v          v
                                     k13        k14
```

### Selection Order

**Ground Points (1-12):**

1. **k1** - Bottom-left corner (baseline left)
2. **k2** - Bottom-right corner (baseline right)
3. **k3** - Service box bottom-left
4. **k4** - Service line center-bottom
5. **k5** - Service box bottom-right
6. **k6** - Service box top-left
7. **k7** - Service box top-right
8. **k8** - Service line center-left
9. **k9** - Court center
10. **k10** - Service line center-right
11. **k11** - Top-left corner (baseline left, far end)
12. **k12** - Top-right corner (baseline right, far end)

**Height Points (13-14):**
13. **k13** - Back wall point above k1 (left side wall reference)
14. **k14** - Back wall point above k2 (right side wall reference)

### Height Point Selection Guidelines

- **Location**: Select points on the back wall of the court
- **Alignment**: k13 should be vertically above k1, k14 above k2
- **Height**: Choose points at 2-3 meters height where wall features are visible
- **Precision**: Use clear visual references (lines, marks, equipment) for consistency

## Technical Implementation

### File Structure

```none
trackers/ball_tracker/court_3d_model.py  # Main 3D court model implementation
improvements/homography_14_keypoints.py  # Enhanced homography calculation
improvements/implementation_examples.py  # Usage examples and validation
analytics/projected_court.py             # Court projection and visualization
```

### Key Components

#### Court3DModel Class

- **Purpose**: Creates 3D spatial mapping between image and world coordinates
- **Input**: 14 keypoints with 3D world coordinates
- **Output**: Projection matrices and vanishing point calculations
- **Features**:
  - Depth vanishing point calculation
  - Height vanishing point calculation
  - 3D-to-2D projection mapping

#### Keypoint Correspondence

```python
# Ground level points (z=0)
keypoint_correspondence = {
    0: [0, 0, 0],                    # k1 - bottom-left
    1: [width, 0, 0],                # k2 - bottom-right
    5: [0, length/2, 0],             # k6 - service box left
    6: [width, length/2, 0],         # k7 - service box right
    10: [0, length, 0],              # k11 - top-left
    11: [width, length, 0],          # k12 - top-right
    
    # Height points (z=height)
    12: [0, 0, height],              # k13 - wall left
    13: [width, 0, height],          # k14 - wall right
}
```

#### Enhanced Features with 14 Points

- **3D Ball Tracking**: Accurate ball position in 3D space
- **Trajectory Analysis**: Parabolic ball path reconstruction
- **Height Validation**: Cross-validation using wall reference points
- **Spatial Analytics**: Volume-based player positioning analysis

## Current Implementation Status

### What's Working (12-Point System)

- ✅ Player detection and tracking
- ✅ Basic ball tracking
- ✅ Court boundary detection
- ✅ 2D court projection
- ✅ Analytics data collection

### What's Disabled (14-Point System)

- ❌ 3D court model (Court3DModel)
- ❌ Enhanced ball trajectory tracking
- ❌ Height-based validation
- ❌ Advanced 3D analytics

### Known Issues with 14-Point System

1. **Matrix Rank Issues**: The projection matrix calculation sometimes fails with insufficient independent points
2. **Tensor Conversion Errors**: Some trackers have compatibility issues with the 3D model
3. **Keypoint Correspondence**: Hardcoded assumptions about court dimensions may not match all videos

## Enabling the 14-Keypoint System

### Step 1: Enable 3D Court Model

In `main.py`, change:

```python
# Temporarily disable 3D court model for 12-keypoint compatibility
# court_model = Court3DModel(keypoints=fixed_keypoints_detection)
court_model = None
```

To:

```python
# Enable 3D court model for 14-keypoint system
court_model = Court3DModel(keypoints=fixed_keypoints_detection)
# court_model = None
```

### Step 2: Select 14 Keypoints

- Follow the 14-point selection guide above
- Ensure precise selection of height points on back wall
- Verify keypoints are saved correctly

### Step 3: Test and Validate

- Check for matrix rank errors
- Validate ball trajectory tracking
- Monitor for tensor conversion issues

## Development Roadmap

### Short Term

- [ ] Debug matrix rank calculation issues
- [ ] Fix tensor conversion errors in player keypoints tracker
- [ ] Add keypoint validation during selection

### Medium Term

- [ ] Implement adaptive court dimension detection
- [ ] Add support for different court orientations
- [ ] Create automated keypoint verification system

### Long Term

- [ ] Machine learning-based keypoint detection
- [ ] Real-time 3D court model calibration
- [ ] Multi-camera 3D reconstruction support

## Troubleshooting

### Common Issues

#### Matrix Rank Error

AssertionError: Matrix A must have rank 12. Choose at least 6 independent points

**Solution**:

- Ensure keypoints are not collinear
- Check that height points are truly at different heights
- Verify court dimensions in configuration

#### Tensor Conversion Error

RuntimeError: a Tensor with 2 elements cannot be converted to Scalar

**Solution**:

- Related to player keypoints tracker
- Currently requires 12-point system
- Future fix needed in tracker implementation

#### Keypoint Selection Issues

- **Problem**: Inaccurate keypoint placement
- **Solution**: Use clear visual references, take time for precision
- **Tool**: Consider adding keypoint validation feedback

## Configuration

### Court Dimensions

```python
# Standard padel court dimensions (meters)
COURT_WIDTH = 10.0    # Court width
COURT_LENGTH = 20.0   # Court length  
COURT_HEIGHT = 3.0    # Wall height for reference points
```

### Model Paths

```python
# Ensure these models support 3D tracking
BALL_TRACKER_MODEL = "path/to/ball_model.pt"
PLAYERS_TRACKER_MODEL = "path/to/players_model.pt"
```

## References

- [Court3DModel Implementation](../trackers/ball_tracker/court_3d_model.py)
- [14-Point Homography](../improvements/homography_14_keypoints.py)
- [Implementation Examples](../improvements/implementation_examples.py)
- [System Architecture](ARCHITECTURE.md)
- [Beginners Guide](BEGINNERS_GUIDE.md)

## Contributing

When working with the 3D court model:

1. **Test with both 12 and 14 keypoint systems**
2. **Validate matrix calculations carefully**
3. **Document any keypoint selection requirements**
4. **Consider backward compatibility with 12-point system**
5. **Add proper error handling for edge cases**

---

*Last updated: July 2025*
*Status: 3D Court Model temporarily disabled for stability*
