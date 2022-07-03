# Anomalous Sound Detection with Pytorch
This repository is a recipe for running the second-place method in Task 2 of the DCASE 2022 competition for the performance of anomalous sound detection systems.  
The method consists of two stages: a feature extractor that utilizes pseudo-anomalous data and an anomalous detector.  
Details of the method are written in [our Technical Report](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Kuroyanagi_11_t2.pdf).

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


## Recipe
- [dcase2022-task2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring): The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd scripts

# Run the recipe from scratch
$ ./job.sh

# You can change config via command line
$ ./job.sh --no <the_number_of_your_customized_yaml_config>

# You can select the stage to start and stop
$ ./job.sh --stage 1 --start_stage 3

# After all machine types have completed Stage 5, starting Stage 2.
# You can see the results at exp/all/**/score*.csv
$ ./job.sh --stage 2

# If you would like to ensemble several models, please following commands.
$ . ./path.sh
$ python ./local/get_domain_classifier_weight.py
$ python ./local/domain_generalization_ave.py

```

## Citation
```
@techreport{KuroyanagiNU-HDL2022,
    Author = "Kuroyanagi, Ibuki and Hayashi, Tomoki and Takeda, Kazuya and Toda, Tomoki",
    title = "Two-stage anomalous sound detection systems using domain generalization and specialization techniques",
    institution = "DCASE2022 Challenge",
    year = "2022",
    month = "July",
}
```


## Author

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
