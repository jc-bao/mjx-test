model_list=("g1_8dof_mjx_sphere.xml" "g1_8dof_mjx_capsule.xml")
batch_size_list=(512 1024 2048 4096 8192 16384)
iterations_list=(1 2 3 4 5 6 7 8 9 10)
ls_iterations_list=(1 2 3 4 5 6 7 8 9 10)
default_batch_size=2048
default_batch_size_cpu=32
default_iterations=2
default_ls_iterations=5
default_model="g1_8dof_mjx_sphere.xml"

# evaluate speed for each model
# for model in $model_list; do
#   python 01testspeed_mjx.py --mjcf=./assets/$model --base_path=. --nstep=1000 --batch_size=1024 --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv
# done

# evaluate speed for each batch size
# for batch_size in $batch_size_list; do
#   echo "evaluating batch size $batch_size"
#   python 01testspeed_mjx.py --mjcf=./assets/g1_8dof_mjx_capsule.xml --base_path=. --nstep=1000 --batch_size=$batch_size --unroll=1 --solver=newton --iterations=2 --ls_iterations=5 --output=csv
# done

# evaluate speed for each iterations number
# for iterations in $iterations_list; do
#   echo "evaluating iterations $iterations"
#   python 01testspeed_mjx.py --mjcf=./assets/$default_model --base_path=. --nstep=1000 --batch_size=$default_batch_size --unroll=1 --solver=newton --iterations=$iterations --ls_iterations=$default_ls_iterations --output=csv
# done

# evaluate speed for each linsearch iterations number
# for ls_iterations in $ls_iterations_list; do
#   echo "evaluating ls_iterations $ls_iterations"
#   python 01testspeed_mjx.py --mjcf=./assets/$default_model --base_path=. --nstep=1000 --batch_size=$default_batch_size --unroll=1 --solver=newton --iterations=$default_iterations --ls_iterations=$ls_iterations --output=csv  
# done

# test cpu mujoco
# add /home/pcy/mambaforge/envs/jax12/lib/python3.12/site-packages/mujoco to LD_LIBRARY_PATH
for model in $model_list; do
    export LD_LIBRARY_PATH=/home/pcy/mambaforge/envs/jax12/lib/python3.12/site-packages/mujoco:$LD_LIBRARY_PATH
    output=$(./02testspeed_mjc "./assets/$model" 1000 $default_batch_size_cpu 0.1 0 | grep -E "Total steps per second|Realtime factor")
    fps=$(echo "$output" | grep "Total steps per second" | awk -F ':' '{gsub(/ /, "", $2); print $2}')
    single_realtime_factor=$(echo "$output" | grep "Realtime factor" | awk -F ':' '{gsub(/ /, "", $2); print $2}' | awk '{print $1}')
    single_realtime_factor="${single_realtime_factor%x}"
    if [ ! -f mjc_speed_test.csv ]; then
        echo "model,fps,single_realtime_factor,batch_size" > mjc_speed_test.csv
    fi
    model_name="${model%.xml}"
    echo "$model_name,$fps,$single_realtime_factor,$default_batch_size_cpu" >> mjc_speed_test.csv
    echo "Data appended to mjc_speed_test.csv: model=$model_name, fps=$fps, single_realtime_factor=$single_realtime_factor"
done