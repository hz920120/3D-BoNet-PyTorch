cd Utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd Utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../