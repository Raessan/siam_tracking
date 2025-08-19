# Single-object tracking using Siamese Neural Network

This repository implements a Siamese Neural Network with cross-attention for single-object tracking on the [LaSOT](https://hengfan2010.github.io/projects/LaSOT/) dataset. It provides scripts for:

- Dataset processing: preparing template–search pairs with photometric augmentations
- Training: optimizing classification heatmap and bounding-box regression heads
- Inference: running the trained model to track objects in new videos

## Installation

1. Clone this repository:
``` 
git clone https://github.com/Raessan/siamese-tracker.git
cd siamese-tracker
```

2. (Optional) Create a Python or conda environment
3. Install project and dependencies
```
pip install -e .
```
Please, note that `torch` and `torchvideo` should be installed according to your CUDA device.

## Usage

### Configuration

The configurable variables are in the `config/config.py` file. This should be set prior to the launch of either the training or the inference. Most of the variables should be self-explained and have default working values.

### Training

With the configured parameters in the `config` file. The `train_tracker.ipynb` can be executed to train the algorithm. Since dinov3 already provides good features, I just trained for one epoch using the default configuration.

### Inference

The inference is performed by executing the `inference.py` file as follows: `python inference.py <video_file_or_camera_index>`. If an integer is provided, inference is performed with the camera corresponding to that index. If a path is provided, the inference is performed with the video. If no extra argument is provided, it defaults to 0.

In both modalities (camera and video), the app requests the user to draw a bounding box around the object to track, with the difference that, in the case of a video, the first frame is frozen until the bounding box is drawn, while with the camera it does not freeze at the beginning.

Then, the program draws the resulting template image, and the upcoming search images, with their bounding box in those search images and also its extrapolation to the original image. Also, the heatmap is drawn to see the classification output.

## Insight into the algorithm

### Dataset processing

The dataset consists of groups of images (each group corresponding to a video) that are labeled with the bounding box of the object that is being tracked in that video. The dataset processing includes getting positive or negative pairs to feed the NN. Each pair is composed of two images:

- The template image is used for reference, which always contains the object.
- The search image may contain an object or not, corresponding to positive or negative examples, respectively.

These images are not the original images from the dataset, but rather a smaller crop of the original image, containing the bounding box and an extra context given by the `extra_context` variables. For the template image, this context is always the same percentage. But for the search image this context is selected randomly between two values, to simulate different sizes of the same object. If the required extra context entails going out of the bounds of the original image, then that image is padded with replicated border. Once cropped, both template and search images are resized to the final size, with possible more photogrametric augmentations before converting them to tensors.

The are two variables that are used as labels for supervision:

- A heatmap for classification, of size `(SIZE_OUT, SIZE_OUT, 1)`, which contains floats between 0 and 1 representing the presence of the object: At the center of the bounding box, the heatmap contains `1` but as we go further from the center, it decays using a tent function until the border of the bounding box, where it turns `0`.

- A bounding box regressor that provides the width and height of the bounding box, and optionally (if `REG_FULL` parameter is `True`) also `x` and `y` offsets to refine the location of the bounding box. In my experiments I tried mainly `REG_FULL = False` since it provides already well centered bounding boxes. 

For negative samples, both the heatmap and the bounding box regressor are set entirely to 0.

### Neural Network

#### 1. Siamese Backbone (shared weights):

- A Dinov3 left untouched (although it is possible to unfreeze some layers).

#### 2. Cross-Attention Module:

- Implements multi-head attention: template features as queries, search features as keys and values.
- Computes attention maps to highlight search regions similar to the template.
- Applies a 1×1 convolution and residual connection to fuse attended features.

#### 3. Feature Fusion & Heads:

- A 1×1 fuse convolution reduces and refines the attended search features.
- Classification Head: 3×3 conv + BatchNorm + ReLU + Dropout, followed by 1×1 conv → single-channel heatmap predicting object presence.
- Regression Head: same structure but outputs 4 channels: (dx, dy, w, h) offsets relative to grid centers.
    - `reg_full=True`: (dx, dy) allowed negative/positive, (w, h) passed through Softplus to ensure positivity.
    - `reg_full=False`: raw (w, h) for scale-only regression.

### Loss function: Focal Classification + Masked Box Regression

The model is supervised with a combination of:

1. Focal Binary Cross-Entropy Loss for classification (heatmap prediction), and
2. Masked L1 Loss for bounding-box regression.

```
total_loss = cls_loss + weight * reg_loss
```

#### Classification Loss (`cls_loss`)

- Uses focal loss to handle class imbalance, since most of the heatmap is background (gt_heat == 0).
- Focal loss dynamically scales the binary cross-entropy:
    - Positive samples (object center) are weighted by:
    α * (1 - p)^γ → focusing more on hard (low-confidence) positives.
    - Negative samples (background) are weighted by:
    (1 - α) * p^γ → reducing easy negatives' influence.
- The total classification loss is normalized by the number of positive samples.

#### Regression Loss (`reg_loss`)

- Applies L1 loss (`|pred - gt|`) on the predicted bounding box (either (w, h) or (dx, dy, w, h)).
- Loss is computed only at locations where the object is present (i.e. gt_heat > 0), using a binary mask.
- Normalized by the number of positive locations to ensure scale invariance across images.

## References

- [Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., & Torr, P. H. (2016). Fully-convolutional siamese networks for object tracking. *In European conference on computer vision*, (pp. 850-865). Cham: Springer International Publishing](https://arxiv.org/abs/1606.09549)

- [Yu, Y., Xiong, Y., Huang, W., & Scott, M. R. (2020). Deformable siamese attention networks for visual object tracking. *In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, (pp. 6728-6737)](https://arxiv.org/abs/2004.06711)

- [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (2025). DINOv3. *arXiv preprint arXiv:2508.10104.*](https://arxiv.org/abs/2508.10104)