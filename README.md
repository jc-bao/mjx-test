# mjx-test

```
# test speed for g1 model
python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_sphere.xml --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv

python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_capsule.xml --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv

```

## Questions

> Why realtime factor will be affected by batch size?

I can understand if the batch size is too large, there would be not enough cuda cores to handle the computation, so the realtime factor will be affected. But why the realtime factor will be affected by batch size when it is not large?

> Why different contact model shows different realtime factor?

In `sphere` model, 