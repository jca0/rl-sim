# mujoco_sim: MuJoCo Scene and Teleoperation

## Files

| File | Purpose |
|------|---------|
| `trs_so_arm100/` | Robot model (URDF, meshes, scene) |
| `teleop.json` | 50 human demonstrations (WORKING) |
| `teleop_dense.json` | Dense demos (NOT working) |
| `keyboard_teleop.py` | Manual keyboard control |
| `playback_action_sequence.py` | Replay demo sequences |
| `mujoco_sim.py` | Basic MuJoCo simulation |
| `mujoco_pick_place.py` | Pick-and-place environment |

## Robot Model (trs_so_arm100/)

- `scene.xml` - Main scene with robot, cube, table
- `so_arm100.xml` - Robot URDF definition
- `assets/` - STL mesh files for each link

**Keyframes:**
- `home_with_cube` - Arm above cube (starting position)
- `rest_with_cube` - Arm folded

## Demonstrations (teleop.json)

50 human-recorded pick-and-place sequences:
- Each sequence: 5-6 waypoints with timing
- Task: Pick red cube, move 5cm left, place
- Duration: ~6 seconds per demo

**Usage:**
```python
import json
with open("mujoco_sim/teleop.json") as f:
    demos = json.load(f)
# demos[0]["waypoints"] = list of {joints: [6], duration: float}
```

## Keyboard Teleoperation

```bash
python mujoco_sim/keyboard_teleop.py
```
Controls: WASD for joints, Space for gripper

