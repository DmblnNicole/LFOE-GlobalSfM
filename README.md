# Learning to Filter Outlier Edges in Global SfM
## [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)

## About

This project extends GLOMAP with an outlier filter for relative translations in the view graph. The filter is a Graph Neural Network that detects and removes incorrect relative poses to improve camera position accuracy. GLOMAP takes a COLMAP database as input. After the view graph is built and global rotations are estimated, the outlier filter is applied to remove noisy translations. The final output is a sparse reconstruction in COLMAP format.


## Build Instructions

1. First, install [COLMAP](https://colmap.github.io/install.html#build-from-source).

2. [GLOMAP](https://github.com/colmap/glomap) is integrated as a submodule in external/glomap:
    ```bash
    git clone git@github.com:DmblnNicole/lfoe.git
    cd lfoe
    git submodule init
    git submodule update
    ```

    You should see external/glomap.

3. Install dependencies:
   
   ```bash
   sudo apt update
   sudo apt install pybind11-dev libyaml-cpp-dev

5. Build the project:
   
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
7. Install Conda environment:
   
    ```bash
    conda create -n lfoe python=3.10
    conda activate lfoe
    pip install -r requirements.txt
    ```
---

## Run

Run GLOMAP with outlier filter from ```/build``` directory:

```bash
./glomap_filter mapper --database_path <path/to/database.db> --output_path <path/to/output/> --image_path <path/to/images/>
```
- --database_path: COLMAP reconstruction
- --output_path: Path to save filtered GLOMAP reconstruction
- --image_path: Path to images

## Visualize

```bash
colmap gui --import_path <path/to/output/> --database_path <path/to/database.db> --image_path <path/to/images/>
```

## Compare with COLMAP reconstruction

```bash
colmap model_comparer --input_path1 <path/to/colmap/reconstruction/> --input_path2 <path/to/output/>
```

## Notes

Currently, filtering uses DINO features, while GLOMAP uses vocabulary tree-based image retrieval. In the future, this could be replaced with DINO-based retrieval to streamline the pipeline.
