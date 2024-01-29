# Prototypical Networks for Few-Shot Learning
(ONGOING) This repository contains an implementation of Prototypical Networks for few-shot learning using PyTorch. Prototypical Networks are a popular approach for few-shot learning tasks, where the goal is to recognize new classes with very limited training examples.

## Requirements
Before running the code, make sure you have the following dependencies installed:

Python 3.x
PyTorch
NumPy
argparse (for command-line arguments)
You can install PyTorch and NumPy using pip:

```bash
conda create -n ProtoNets python=3.11
pip install -r requirements.txt
```
Usage
Data Preparation
Organize your dataset in a CSV file with the following structure:

```
filename,label
image_name_1.jpg,class1
image_name_2.jpg,class2
```
Specify the path to the root directory in which are the csv file and a folder "images" which contains the images.

## Training
To train the Prototypical Network, you can use the following command:

bash
```
python train.py --csv_file your_dataset.csv --root_dir /path/to/your/dataset --n_ways 5 --k_shots 5 --m_queries 5 --num_episodes 10 --tasks_per_episodes 100 --metric euclidean --channels 3 --embedding_dim 256 --lr 0.001
```
For which the arguments are:
```
--csv_file: Path to the CSV file containing dataset information.
--root_dir: Root directory of the dataset.
--n_ways: Number of classes in each few-shot task.
--k_shots: Number of support examples per class.
--m_queries: Number of query examples per class.
--num_episodes: Number of training episodes.
--tasks_per_episodes: Number of tasks sampled per episode.
--metric: Distance metric for computing prototypes ('euclidean' or 'cosine').
--channels: Number of input image channels (e.g., 3 for RGB).
--embedding_dim: Dimension of the embedding produced by the encoder.
--lr: Learning rate for training.
```
## Results
After training, the model's performance on few-shot learning tasks will be printed. You can further evaluate the model on your own few-shot tasks using the trained model checkpoint.

## Acknowledgments
This implementation is based on the original paper: Prototypical Networks for Few-shot Learning by Jake Snell, Kevin Swersky, and Richard Zemel.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Author:
Hans-Olivier Fontaine
