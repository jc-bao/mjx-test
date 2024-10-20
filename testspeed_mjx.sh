model_list=("g1_8dof_mjx_sphere.xml" "g1_8dof_mjx_capsule.xml")
batch_size_list=(512 1024 2048 4096 8192 16384)

# evaluate speed for each model
# for model in $model_list; do
#   python 01testspeed_mjx.py --mjcf=./assets/$model --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv
# done

# evaluate speed for each batch size
for batch_size in $batch_size_list; do
  echo "evaluating batch size $batch_size"
  python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_sphere.xml --base_path=. --nstep=1000 --batch_size=$batch_size --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv
done