To download the dataset:
```
bash dowload_dataset.sh
```
This will download and unzip the dataset in the data folder and will unzip pre-trained models.
The data directory has the following organization.

```
├── shapes
│   ├── all_ids.txt
│   ├── face_data_release.zip
│   ├── meshes.zip
│   ├── test_data.h5
│   ├── test_ids.txt
│   ├── train_data.h5
│   ├── train_ids.txt
│   ├── val_data.h5
│   └── val_ids.txt
└── spline
    ├── closed_splines.h5
    ├── open_splines.h5
    └── simple_less_thn_20.zip
```

* `meshes.zip`: contains all the meshes used in the parsenet
  experiments.  *Note* that these models are taken from ABC dataset. We
  pre-processed shapes to separate disconnected meshes into different
  meshes. For that reason you will notices that names of the shapes is
  of the format `shapeid_index.json`, where `shapeid` is the id of the
  model from ABC dataset and `index` is the index of the disconnected
  part. 
  
* `train_data.h5`, `train_data.h5` and `val_data.h5`: contain points,
  normals, segment index and primitive type index for each shape.
  Please refer to `src/dataset_segments.py` on how to load these h5
  files. Note that, for primitive types, there are possible 10 primitives,
  for example circle, sphere, plane, cone, cylinder, open spline, closed spline,
  revolution, extrusion and `extra`. revolution, extrusion and extra are treated
  as b-spline primitives because b-spline can also approximate these patches. Excluding
  shapes with these extra surface patches would have resulted in very small dataset.
  More specifically:
  1. [0, 6, 7, 9] indices correspond to closed b-spline.
  2. [2, 8] indices correspond to open b-spline.
  3. [1] index corresponds to plane.
  4. [3] corresponds to cone.
  5. [4] corresponds to cylinder.
  6. [5] corresponds to sphere.
  
* `face_data_release.zip`: contains txt files for each shape in the
  above dataset. Specifically, it contains the segment id and
  primitive types for each shape.

* `train_ids.txt`, `val_ids.txt` and `test_ids.txt`: contains shape ids for
  different splits. `all_ids.txt` contains list of ids for all shapes.

* `closed_splines.h5`: contains points and control points for closed
  splines. Please refer to `src/dataset.py` for more details on how to
  load points, and splits. `open_splines.h5` is for open splines.
