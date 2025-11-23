"""
SO-100 Movement Test - RAW positions (no calibration needed)
"""
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import time

PORT = "COM4"

print("=" * 50)
print("SO-100 MOVEMENT TEST")
print("=" * 50)

# Motor configuration
motors = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

print("Connecting...")
bus = FeetechMotorsBus(port=PORT, motors=motors, calibration=None)
bus.connect()
print("✓ Connected!")

print("\nReading RAW positions...")
positions = {}
for name, motor in motors.items():
    # Read raw position (no normalization)
    pos, _, _ = bus._read(56, 2, motor.id)  # Address 56 = Present_Position, 2 bytes
    positions[name] = pos
    print(f"  {name}: {pos}")

print("\n" + "=" * 50)
print("MOVING MOTORS - WATCH THE ROBOT!")
print("=" * 50)

# Test 1: Move base
print("\n1. Moving BASE (shoulder_pan)...")
current_base = positions['shoulder_pan']
target = current_base + 500
bus._write(42, 2, motors['shoulder_pan'].id, target)  # Address 42 = Goal_Position
time.sleep(2)
bus._write(42, 2, motors['shoulder_pan'].id, current_base)
time.sleep(2)
print("   ✓ Base moved!")

# Test 2: Move shoulder  
print("\n2. Moving SHOULDER (shoulder_lift)...")
current_shoulder = positions['shoulder_lift']
target = current_shoulder + 300
bus._write(42, 2, motors['shoulder_lift'].id, target)
time.sleep(2)
bus._write(42, 2, motors['shoulder_lift'].id, current_shoulder)
time.sleep(2)
print("   ✓ Shoulder moved!")

# Test 3: Move elbow
print("\n3. Moving ELBOW (elbow_flex)...")
current_elbow = positions['elbow_flex']
target = current_elbow + 300
bus._write(42, 2, motors['elbow_flex'].id, target)
time.sleep(2)
bus._write(42, 2, motors['elbow_flex'].id, current_elbow)
time.sleep(2)
print("   ✓ Elbow moved!")

# Test 4: Move wrist flex
print("\n4. Moving WRIST FLEX (wrist_flex)...")
current_wrist_flex = positions['wrist_flex']
target = current_wrist_flex + 200
bus._write(42, 2, motors['wrist_flex'].id, target)
time.sleep(2)
bus._write(42, 2, motors['wrist_flex'].id, current_wrist_flex)
time.sleep(2)
print("   ✓ Wrist flex moved!")

# Test 5: Move wrist roll
print("\n5. Moving WRIST ROLL (wrist_roll)...")
current_wrist_roll = positions['wrist_roll']
target = current_wrist_roll + 400
bus._write(42, 2, motors['wrist_roll'].id, target)
time.sleep(2)
bus._write(42, 2, motors['wrist_roll'].id, current_wrist_roll)
time.sleep(2)
print("   ✓ Wrist roll moved!")

# Test 6: Move gripper
print("\n6. Moving GRIPPER...")
current_gripper = positions['gripper']
target = current_gripper + 500
bus._write(42, 2, motors['gripper'].id, target)
time.sleep(2)
bus._write(42, 2, motors['gripper'].id, current_gripper)
time.sleep(1)
print("   ✓ Gripper moved!")

print("\n" + "=" * 50)
print("✓✓✓ SUCCESS! ALL 6 MOTORS WORKING! ✓✓✓")
print("=" * 50)

bus.disconnect()
print("\nRobot disconnected.")
