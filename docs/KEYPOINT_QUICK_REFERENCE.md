# 🎯 Quick Reference: Keypoint Selection Systems

## Current System: 12-Keypoint (Active)

### When to Use

- ✅ **Default choice** for all current analysis
- ✅ Stable and fully tested
- ✅ Works with all trackers
- ✅ Sufficient for most analytics needs

### Selection Order

```none
k11─────────────────k12
│                    │
k8──────k9──────────k10
│       │            │
k6──────────────────k7
│       │            │
k3──────k4──────────k5
│                    │
k1──────────────────k2
```

Click in order: k1 → k2 → k3 → k4 → k5 → k6 → k7 → k8 → k9 → k10 → k11 → k12

## Advanced System: 14-Keypoint (Disabled)

### When to Use

- 🔬 **Advanced development** and research
- 🔬 **3D ball trajectory** analysis needed
- 🔬 **Enhanced spatial** accuracy required
- ⚠️ **Currently disabled** due to stability issues

### Additional Points

- **k13**: Back wall point above k1 (left side)
- **k14**: Back wall point above k2 (right side)

### Enabling (For Developers)

1. Edit `main.py` line 110-113:

   ```python
   # Change this:
   court_model = None
   
   # To this:
   court_model = Court3DModel(keypoints=fixed_keypoints_detection)
   ```

2. Select 14 keypoints instead of 12
3. Test for matrix rank and tensor conversion errors

## 🚨 Current Status

| Feature | 12-Point | 14-Point |
|---------|----------|----------|
| **Status** | ✅ Active | ❌ Disabled |
| **Stability** | ✅ Stable | ⚠️ Issues |
| **Player Tracking** | ✅ Full | ✅ Full |
| **Ball Tracking** | ✅ 2D | 🔬 3D |
| **Court Projection** | ✅ 2D | 🔬 3D |
| **Recommended For** | Production | Development |

## 📚 Documentation Links

- **Full Guide**: [3D Court Model & 14-Keypoint Selection](3D_COURT_MODEL_AND_14_KEYPOINTS.md)
- **Beginner Tutorial**: [Beginners Guide](BEGINNERS_GUIDE.md)
- **Architecture**: [Architecture Overview](ARCHITECTURE.md)

---
*Quick tip: Stick with 12-keypoint system unless you specifically need 3D features*
