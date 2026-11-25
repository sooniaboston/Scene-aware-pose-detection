# Dataset splits used in the paper

The images themselves are **not** redistributed (see the Data
availability statement in the paper). These are the exact file lists
and labels of the subsets that were used.

| subset | images | climb | no-climb |
|---|---|---|---|
| `train1` | 500 | 371 | 129 |
| `test1` | 50 | 35 | 15 |
| `folders_pool` | 200 | 100 | 100 |

## Label semantics

`label = 1` means *climb*, `label = 0` means *no-climb*.
The columns `cx, cy, w, h` are the YOLO-normalised person bounding
box of the labelled instance.
