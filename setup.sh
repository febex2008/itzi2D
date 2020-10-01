# sudo rm -R build
# sudo -H pip uninstall itzi
# sudo cython  itzi/flow.pyx
# sudo cython  itzi/swmm/swmm_c.pyx
# find . -name "*.pyc" -exec rm -rf {} \;
# sudo -H python -m pip install -e .

sudo rm -rf build/ 
sudo rm -rf itzi.egg-info/
python -m pip install cython
python -m pip install pyinstrument numpy networkx==1.11
sudo pip uninstall itzi
python -m cython -f itzi/swmm/swmm_c.pyx itzi/flow.pyx
python -m pip install -e .
