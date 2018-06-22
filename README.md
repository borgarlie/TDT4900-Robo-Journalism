# TDT4900 - Computer Science, Master's Thesis
Improving Abstractive Text Summarization Models by using Reinforcement Learning and Generative Adversarial Networks

This repository contains the code used to produce the results seen in [tdt4900_final_delivery.pdf](tdt4900_final_delivery.pdf)

## Installing and running the project

1. Clone this github repo.

2. Install PyTorch. In this project we use python 3.6.1 with cuda:

```sh
$ pip3 install torch torchvision
```
Alternative: download the wheel file from [pytorch](http://pytorch.org).

3. Install all the required python dependencies using pip3.

```sh
$ pip3 install -r /path/to/requirements.txt
```

## Execution

Running generator pretraining:
```sh
$ python3 training/seq2seq/run_experiment.py path/to/config/folder 0
```

```sh
Usage: run_experiment.py
 	[Config folder path]
	[Cuda device]
```

Running discriminator pretraining:
```sh
$ python3 training/classifier/run_experiment.py path/to/config/folder 0
```

```sh
Usage: run_experiment.py
 	[Config folder path]
	[Cuda device]
```

Running RL / GAN training:
```sh
$ python3 training/GAN/run_experiment.py path/to/config/folder 0
```

```sh
Usage: run_experiment.py
 	[Config folder path]
	[Cuda device]
```

Generate fake data to train discriminator:
```sh
$ python3 evaluation/seq2seq/generate_fake_sampled_data.py 0
```

```sh
Usage: generate_fake_sampled_data.py
	[Cuda device]
```

Generate real data to train discriminator:
```sh
$ python3 evaluation/seq2seq/generate_real_data.py
```

Evaluating a model (generating summaries):
```sh
$ python3 evaluation/seq2seq/evaluate_test_data.py 0
```

```sh
Usage: evaluate_test_data.py
	[Cuda device]
```

## Preprocessing and dataset

The tokenized version of the CNN/Daily Mail dataset can be downloaded from [Dataset Link](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).

[preprocess_cnn.py](preprocess/preprocess_cnn.py) can be run to separate articles from the ground truths and at the same time restrict the length to a minimum and a maximum length. At this point, the new preprocessed dataset should consist of two files. One with an article per line and one with the corresponding reference summary per line.  Before running, preprocess_cnn.py needs to be altered to fit relative paths to load the downloaded tokenized dataset and save the preprocessed dataset.

Further, [preprocess_pointer.py](preprocess/preprocess_pointer.py) can be run to create a pickle file with the final preprocessed dataset and the vocabulary.  Before running, preprocess_pointer.py needs to be altered to fit relative paths to load the dataset preprocessed in the previous step and save the final preprocessed dataset.

## Configuration

The experiment/config folder for each experiment (preprocessing generator, preprocessing discriminator and training with RL / GAN) should contain a file named `config.json`. Example for RL / GAN training:

```json
{
  "train" : {
    "dataset" : "../../data/cnn_pickled/cnn_pointer_50k",
    "num_articles" : -1,
    "num_evaluate" : 13000,
    "throw" : 0,
    "n_epochs" : 5,
    "batch_size" : 50,
    "discriminator_batch_size" : 32,
    "generator_learning_rate" : 0.0001,
    "discriminator_learning_rate" : 0.0001,
    "beta" : 0.9984,
    "lambda": 0.5,
    "n_generator" : 4,
    "n_discriminator" : 16,
    "discriminator_n_epochs" : 4,
    "discriminator_fake_data_sample_rate" : 1.00,
    "num_monte_carlo_samples" : 8,
    "max_sample_length" : 99,
    "sample_rate" : 0.20,
    "allow_negative_reward" : true,
    "use_trigram_check" : false,
    "use_running_avg_baseline": true
  },
  "evaluate" : {
    "expansions" : 3,
    "keep_beams" : 6,
    "return_beams": 3
  },
  "generator_model" : {
    "embedding_size" : 100,
    "n_layers" : 1,
    "hidden_size" : 128,
    "dropout_p" : 0.0,
    "load" : true,
    "load_file" : "../../models/pretrained_models/generator/generator_pretrained_epoch13.pth.tar"
  },
  "discriminator_model" : {
    "hidden_size" : 128,
    "dropout_p" : 0.5,
    "num_kernels" : 100,
    "kernel_sizes" : [3, 4, 5],
    "load" : true,
    "load_file" : "../../models/pretrained_models/classifier/cnn/discriminator_pretrained_epoch50.pth.tar"
  },
  "save" : {
    "save_file_generator" : "generator_gan_test_1.pth.tar",
    "save_file_discriminator" : "discriminator_gan_test_1.pth.tar"
  },
  "log" : {
    "print_every" : 200,
    "filename": "experiments/cnn_test_1/output_gan_test_1.log"
  },
  "tensorboard" : {
    "log_path" : "../../log/GAN/gan_test_1"
  }
}

```

An example is provided [here](training/GAN/experiments/cnn_test_1/config.json).

Explaination of some of the fields:

`dataset`: Relative path to the pickled dataset, without the .pickle suffix.

`num_articles`: Number of articles to include from the dataset, -1 means the whole file.

`num_evaluate`: Number of articles used in the validation and evaluation set.

`throw`: Number of articles to not include in either the train or the evaluation set.

`generator_model/load_file`: Relative path to the saved generator.

`discriminator_model/load_file`: Relative path to the saved discriminator.

Other fields in the config are default values, and not all are relevant. Those that are most relevant to play around with are the model parameters, e.g. `hidden_size`, `embedding_size` and `dropout_p`.


When training with RL and GAN there are a few details not included in the config file. E.g. which roll-out strategy to use and which objective function to use. For instance, to use the naive roll-out strategy, comment out the other strategies and only include `generator = GeneratorRlStrat(...)`. To use the ROUGE objective function, comment out the other `discriminator` alternatives and only include `discriminator = RougeDiscriminator(...)`.


## Results

### Example article:

bayern munich star david alaba suffered a knee ligament injury in their 2-0 victory over as roma in the champions league on wednesday . the gifted 22-year-old austria international was in outstanding form , setting up the first goal for franck ribery . david alaba had to be taken off in the second half of bayern ’s 2-0 win over roma on wednesday night . the austrian international had set up the opening goal for franck ribery at the allianz arena . he was taken off injured in the 81st minute as bayern cruised to the champions league knockout stage after their win over the italians secured them top spot in group e. ‘ there is a problem with david . he was injured , ’ bayern coach pep guardiola told reporters . ‘ he will get checked out tomorrow when we will know more . ’ the club later said on twitter the player sustained ’ a medial collateral ligament injury ’ . bayern were already without a string of key injured midfielders including bastian schweinsteiger , javi martinez and thiago alcantara . alaba joins a growing midfield injury list for the german champions .

### Example generated summary:

david alaba had to be taken off in the second half of bayern ’s 2-0 win over roma in the champions league on wednesday night . the austrian international had set up the opening goal for franck ribery . the club later said on twitter the player sustained ’ a medial collateral ligament injury ’

Results from all the best models (as shown in the thesis) is found [here](results).

## Notes

This repository contains a lot of code in a lot of different files used to preprocess the data, run experiments and generate the results shown in the thesis. Here, in this README, we have briefly explained how the preprocessing works and how to train and evaluate the models. Generating other results, such as ROUGE scores are out of the scope for this README.
