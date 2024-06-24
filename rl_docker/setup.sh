#!/bin/bash
set -e
set -u

pip install -e ./IsaacGym_Preview_4_Package/isaacgym/python
pip install -e ./rsl_rl
pip install -e ./legged_gym

pip install numpy==1.23.5
pip install tensorrt==8.5.2.2
/bin/bash
