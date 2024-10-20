# mjx-test

```
# test speed for g1 model
python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_sphere.xml --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv

python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_capsule.xml --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv

```
