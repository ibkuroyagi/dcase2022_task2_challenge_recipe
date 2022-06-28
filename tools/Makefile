DOT:= .
.PHONY: all clean
all: virtualenv
virtualenv:
	./setup_anaconda.sh venv asd_env 3.9
	. ./activate_python.sh; conda update -n base -c defaults conda
	git clone https://github.com/rwightman/pytorch-image-models.git
	. ./activate_python.sh; cd pytorch-image-models; pip install .
	. ./activate_python.sh; cd ../; pip install -e .
	. ./activate_python.sh; conda install ipython ipykernel numpy pandas scikit-learn matplotlib librosa optuna h5py pyyaml tensorboardx black flake8 -c conda-forge
	. ./activate_python.sh; conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch
clean:
	rm -fr venv miniconda.sh activate_python.sh pytorch-image-models
	find -iname "*.pyc" -delete