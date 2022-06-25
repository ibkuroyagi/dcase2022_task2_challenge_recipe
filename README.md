# Anomalous Sound Detection with Pytorch

## Requirements
- Python 3.9+
- Cuda 11.3



## Setup
Note that your cuda version must be exactly matched with the version used for the pytorch binary to install apex.  
To install pytorch compiled with different cuda version, see `tools/Makefile`.  

```bash
$ git clone https://github.com/ibkuroyagi/dcase2022_task2.git
$ cd dcase2022_task2/tools
$ make
```

Note that we specify cuda version used to compile pytorch wheel.  
If you want to use different cuda version, please check `tools/Makefile` to change the pytorch wheel to be installed.

## Recipe
- [dcase2022-task2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring): The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd egs/dcase2022-task2/baseline

# Run the recipe from scratch
$ ./run.sh

# You can change config via command line
$ ./run.sh --conf <your_customized_yaml_config>

# You can select the stage to start and stop
$ ./run.sh --stage 2 --stop_stage 2
```


## Author

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
