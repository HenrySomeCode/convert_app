# Getting Started

## Installation

```
conda create -n convert_app python=3.7
conda activate convert_app
pip install --upgrade setuptools numpy cython lap
pip install --upgrade torch torchvision
pip install future-fstrings mathutils==2.81.2
pip install streamlit==1.23.1
pip install click==8.1.7
pip install protobuf==4.24.4
```

Then, please follow the instruction at https://github.com/TylerGubala/blenderpy/releases to install the bpy. For example, for ubuntu users, please first download the [bpy .whl package](https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl) and then install via:

```
pip install /path/to/downloaded/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
```

## Run

```
streamlit run tools/app_stml.py
```

### Docker

```
docker compose build
docker compose up
```

## Note

py files that use bpy must stand in a same level directory, i.e:

```
root
| tools
| |_convert_pkl2fbx.py
|_streamlit_app.py
```

```
#streamlit_app.py
from tools.convert_pkl2fbx import ConvertPkl2Fbx

```

wouldn't work. This would work:

```
root
|_convert_pkl2fbx.py
|_streamlit_app.py
```

```
#streamlit_app.py

from convert_pkl2fbx import ConvertPkl2Fbx

```
