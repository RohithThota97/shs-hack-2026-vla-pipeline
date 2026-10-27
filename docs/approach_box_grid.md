# Our approach: kinematics-based data augmentation (box grid)

Written by Quynh Nguyen for our team repo ([yugalrk/Robotics--VLA-training](https://github.com/yugalrk/Robotics--VLA-training),
README, 2026-10-27), lightly edited for wording; the content is hers. The notes at the end say where each step
lives in this pipeline.

---

Every one of the 84 handover demonstrations is the same scripted motion; only the lighting changes. This showed us
our main challenge: the training data has very little diversity, which is why we looked for ways to generate more
varied demonstrations.

**Our approach: kinematics-based data augmentation.** We reuse recorded robot movements and combine them with the
robot's kinematic model to calculate new trajectories for different box positions, without training an additional
AI model.

## 1. Data exploration

We analyse the existing demonstrations: camera recordings, joint states and actions. Each robot state has 16
values, seven joint angles and one gripper state per arm. These recordings are the reference movement patterns for
generating new trajectories.

## 2. Robot kinematics and trajectory reconstruction

We combine the recorded joint angles with the FR3 Duo's kinematic model, which describes its link geometry, joint
axes and movement limits. Two main algorithms are used:

- **Forward kinematics (FK)** rebuilds the robot's original gripper positions and paths in 3D space from the recorded
  joint angles.
- **Inverse kinematics (IK)** works out the joint angles needed to reach new target positions.

Our IK implementation uses damped least squares (DLS) with joint-limit constraints and null-space optimisation, so
the computed joint angles stay close to the original recorded movements. This lets us estimate the robot's reachable
workspace and adapt movements to new box positions while preserving the reference trajectory as much as possible.

## 3. Trajectory retargeting and lookup table

We divide the tabletop into a 5 cm grid and calculate candidate left-arm trajectories for different box positions:

- the box position seen by the camera is mapped to a tabletop grid cell;
- the original gripper path is adapted to the new target;
- inverse kinematics calculates the matching joint movements;
- the resulting trajectories are precomputed and stored in a lookup table (LUT) for fast retrieval.

Results so far:

- 136 reachable grid positions, each with a candidate left-arm trajectory;
- 26 grid positions outside the reachable workspace under the implemented constraints;
- an interactive view of the computed trajectories for each box position.

## 4. Towards automated training data generation

The next step is to run and check the adapted trajectories in Isaac Sim, including grasp success, collision checks
and the complete dual-arm handover. Successful runs could then be recorded with synchronised camera images, robot
states and actions, giving new training demonstrations that cover more box positions. These augmented
demonstrations could be used to fine-tune imitation-learning models and to test whether more diverse data improves
generalisation.

---

## Where this lives in the pipeline

| Step | Here |
|---|---|
| 1 Data exploration | Data page; `hackpipe pages site` (page 1: the joint matrix next to the video) |
| 2 FK / IK | `evaluation/kinematics.py` (Franka DH, DLS IK with null space); `sim/scene.py` (both arms' base poses from the scene file, room coordinates); `hackpipe pages site` (page 2) |
| 3 Grid + LUT | `improve/grid.py` (whole cart top at 5 cm, cells `ok` / `out_of_reach` / `off_cart`); `hackpipe pages grid` (the player, with camera → grid cell) |
| 4 Isaac Sim → training | the improve loop (`hackpipe loop run`, Improve page): `record_demos.py --mode grid` records each valid cell, then import + QC → new dataset version → fine-tune → test → compare |

Our numbers on the handover demo (episode 0) at 5 cm: 220 cells, 133 ok, 29 out of reach, 58 off the cart. The 162
cells on the cart match the team's count. Three fewer are ok because we use the FR3 datasheet's joint limits with a
0.05 rad margin, where the team used the scene's slightly wider limits.
