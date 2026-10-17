# Cassava Leaf Disease Classification

A computer vision project that classifies cassava leaf photographs into five disease and health categories. The notebook compares a small convolutional neural network (CNN) baseline with an ImageNet-pretrained EfficientNetB0 model, then evaluates the fine-tuned model and visualizes its image regions with Grad-CAM.

## Dataset

The project uses the labeled training images from Kaggle's Cassava Leaf Disease Classification competition:

[Open the Cassava Leaf Disease Classification dataset on Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

The notebook reads `train.csv`, `train_images/`, and `label_num_to_disease_map.json`. Its label counts are:

| Label | Class | Images |
|---:|---|---:|
| 0 | Cassava Bacterial Blight (CBB) | 1,087 |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 |
| 2 | Cassava Green Mottle (CGM) | 2,386 |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 |
| 4 | Healthy | 2,577 |

The classes are imbalanced, with CMD making up the largest share of the images. The notebook uses a stratified 80/20 train/validation split and class weights during training to give less common classes more influence.

## Approach

1. Explore the labels and visualize the class distribution and sample images.
2. Decode and resize images to 224 × 224 pixels; load them in batches with TensorFlow `tf.data`.
3. Train a small CNN as a baseline.
4. Train EfficientNetB0 with ImageNet weights. Apply random flips, rotations, and zooms during training; first train the classification head with the backbone frozen, then fine-tune its final 30 layers at a lower learning rate.
5. Evaluate with accuracy, a per-class classification report, and normalized confusion matrices.
6. Generate Grad-CAM overlays for validation images to inspect which image regions contribute to predictions.

## Validation results

Results below are from the notebook's held-out validation set of 4,280 images. The fine-tuned EfficientNetB0 improves both overall accuracy and macro-averaged metrics over the baseline.

| Model | Accuracy | Macro recall | Macro F1 |
|---|---:|---:|---:|
| Small CNN baseline | 51.2% | 42.2% | 38.9% |
| EfficientNetB0, frozen backbone | 59.8% | 58.5% | 49.8% |
| EfficientNetB0, fine-tuned | **64.6%** | **61.8%** | **54.3%** |

Fine-tuned EfficientNetB0 validation recall by class:

| Class | Recall |
|---|---:|
| Cassava Bacterial Blight (CBB) | 60.8% |
| Cassava Brown Streak Disease (CBSD) | 54.8% |
| Cassava Green Mottle (CGM) | 59.3% |
| Cassava Mosaic Disease (CMD) | 67.0% |
| Healthy | 66.9% |

Accuracy alone can hide performance differences between classes, so the notebook also reports per-class precision, recall, and F1, alongside confusion matrices. Grad-CAM heatmaps are visual aids for model inspection; they do not establish that a highlighted region caused a prediction.

## Run the notebook

1. Open `cassava-leaf-disease-classification.ipynb` in Kaggle (or upload it to a Kaggle Notebook).
2. Add the competition dataset using the Kaggle link above.
3. Select a GPU accelerator if available.
4. Run the notebook cells from top to bottom. The notebook expects the data under `/kaggle/input/competitions/cassava-leaf-disease-classification`.

The experiment uses TensorFlow/Keras, NumPy, pandas, Matplotlib, Seaborn, and scikit-learn. It sets the random seed to 42 for the split and model setup.

## Notes

- These metrics are from one stratified validation split, not the competition's hidden test set or a leaderboard submission.
- The dataset has a pronounced class imbalance; the reported validation performance may not represent performance on images from new farms, cameras, or growing conditions.
- The notebook creates the evaluation plots and Grad-CAM visualizations during execution.
