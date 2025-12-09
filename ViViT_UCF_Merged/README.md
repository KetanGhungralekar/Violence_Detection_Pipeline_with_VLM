# Guide for the codes in this folder

- The codes named `est___.py` are codes used to estimate the number of clips that will be generated from the original videos.

- The codes named `video_clips_counter.py` are codes used to count the number of clips that are generated from the original videos.

- `find_missing_norm.py` is to find which codes were not normalized - made to resovle an issue when I first tried normalizing, and some videos were left out.

- `hf_vivit.py` is the main code used to finetune the ViViT model on the UCF Crime dataset, using the Hugging Face library.

- `plot__.py` codes are to plot the distribution of the dataset.

- `video_analyzer.py` is to analyze the videos in the dataset.

- `video_clips_counter.py` is to count the number of clips that are generated from the original videos.

- `video_clips.py` is to generate the clips from the original videos.

- `norm__.py` codes are for normalizing the clips.

- The codes in the `prev_model` dir are the final vivit codes used for training.

- `kinetics.py` is the main train, inference and evaluation code.

- `vivit.py` is the class definition of ViViT using `module.py`.

- `module.py` contains the definitions of the different layers in the vivit architecture.

- `dist_augmented_data` directory contains the plots for the final data distribution after augmentation. 