# Feature glossary

One line per family of coordinates of the interaction vector f.
Every coordinate has a one-line verbal gloss, which is what makes
the decision layer auditable.

## `pose_*`

Body-only kinematics, computed from the COCO-17 keypoints alone. This sub-vector is the input of the pose-only baseline.

## `scene_*`

Human-scene interaction predicates: the relation between the body and the obstacle/ground parsed from the scene.
