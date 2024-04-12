# Foundation models of computer vision for river detection

The objective of this project issue the foundation models learnt on natural images such as the recent Segment Anything Model (SAM) by Faceboook and exploit them on the downstream task of **river detection** using the [Dataset](https://zenodo.org/records/8314175) S1S2 Water.

In this project, we implemented two methods using SAM : one **unsupervised method** exploiting prompts, while the other entailed **fine-tuning SAM** on our dataset.\\

## I- Set up

### Step1 : load dataset

Download a part of the dataset through this link : [Dataset](https://zenodo.org/records/8314175). For example we used **part_1**.

### Step2: create a split folder

Using the folder s1s2_water in our repository that is from the [github](https://github.com/MWieland/s1s2_water/tree/main).

- To create a split, configure your split through the file `settings.toml`. Choose your `tile_shape`, enter the path where your data (from Step 1) is located, and the saving directory.

- You need to change catalog.json of your downloaded data : only keep the scenes you have in your subset (downloaded in step 1)
  Run the following command :

  ```
  python s1s2_water/s1s2_water.py --settings s1s2_water/settings.toml
  ```

## II- Code

You can refer to our project report to understand the different methods used and their results.

### Zero shot learning

You can play with our notebooks s1_pipeline.ipynb and s2_pipeline.ipynb contains our zero shot learning pipeline with prompt engineering and sam inference.

### Fine Tune SAM

### Step 1 : Train file

- To train sam you run :
  ```
  python src/train.py --split_path <path_to_split> --ndwi True
  ```

or with slurm

```
  sbatch src/eval_job.sh
```

### Step 2: Eval file

You can evaluate your model using :

```
  python src/eval.py --checkpoint_path <path_to_checkpoint> --split_path <path_to_split> --ndwi True
```

or with slurm

```
  sbatch src/train_job.sh
```
