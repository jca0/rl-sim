~~1. get joint positions from mujoco~~
2. map sim joint positions to irl joint positions
3. run on real robot

[1 tool called]

Use the “rest” keyframe as your shared reference:

**sim qpos:** {'Rotation': 8.554143758417244e-10, 'Pitch': -3.32002662172713, 'Elbow': 3.118134788813849, 'Wrist_Pitch': 1.1813556248365669, 'Wrist_Roll': 3.900751847099726e-06, 'Jaw': -0.17400093667795333}

1. **Sim side:** Reset to `rest_with_cube` (or `rest`) so the MuJoCo joints take the `qpos` values `[0, -3.32, 3.11, 1.18, 0, -0.174]`. Record `sim_qpos = data.qpos[:6]`.

2. **Robot side:** Move the physical SO-100 into its mechanical rest pose (same one used when capturing that keyframe). Read each joint encoder value (`θ_real_rest`) in the units your controller understands.

3. **Solve offsets/scales/directions:**
   - If the hardware reports radians, you can often set `scale=1`. Otherwise set `scale = (hardware_unit_per_radian)`.
   - Offset per joint: `offset = θ_real_rest - direction * scale * θ_sim_rest`. Use `direction = +1` unless you observe that increasing a simulator angle decreases the hardware reading; then set it to `-1`.
   - Check a second pose (e.g., move one joint slightly) to verify the resulting mapping matches; adjust `scale` if needed.

4. **Load into the env:**
```python
offsets = np.array([offset_rot, offset_pitch, ...])
scales = np.array([scale_rot, ...])          # often all 1 if both in radians
directions = np.array([1, -1, ...])          # set to match motor polarity
env.set_joint_calibration(offsets, scales, directions)
```
Or pass them during construction with `joint_offsets=..., joint_scales=..., joint_directions=...`.

5. **Validate:** Convert simulator rest pose via `env.sim_to_real_joint_positions(sim_qpos)` and send those values to the robot; it should already be at rest, so no motion occurs. Then jog each joint a bit, read hardware values, run `env.real_to_sim_joint_positions`—they should align with the MuJoCo angles.

By anchoring both systems at the rest pose you remove ambiguity about zero/offset, ensuring subsequent poses map cleanly.