### Image Segmentation Experiential Learning

This repository contains code for an image segmentation project focusing on segmenting cat images to accurately identify and separate cats from their backgrounds. The project explores various techniques including basic image processing, thresholding, clustering algorithms, and deep learning-based segmentation.

---

#### Dataset Exploration:

The project utilizes the Oxford-IIIT Pet Dataset, which consists of 37 categories of pet images with associated ground truth annotations for breed, head ROI, and pixel level trimap segmentation.

---

#### Hands-on Basic Image Processing:

The code demonstrates basic image processing techniques including:
- Converting images to grayscale
- Applying Gaussian blur
- Performing edge detection
- Different thresholding techniques such as binary, binary inverse, truncated, to zero, to zero inverse, Otsu's, and adaptive thresholding.

---

#### Clustering Algorithms:

Clustering algorithms, specifically K-means clustering, are applied to segment images from the dataset. The effectiveness of the algorithm is evaluated through visualization.

---

#### Deep Learning Model:

A pre-trained deep learning model based on the U-Net architecture is utilized for image segmentation. The model employs a MobileNetV2 encoder and upsample blocks for the decoder. The dataset is augmented and trained using TensorFlow.

---

#### Repository Structure:

- **image-segmentation.ipynb**: Jupyter notebook containing the code for the image segmentation project.
- **README.md**: Markdown file containing project overview, instructions, and outputs.


---

#### How to Use:

1. Clone the repository to your local machine.
2. Open the `Code.ipynb` notebook using Jupyter or Google Colab.
3. Follow the instructions provided in the notebook to run the code and explore the results.
4. Experiment with different parameters and techniques for image segmentation.

---

## Dataset and Citations

- The code utilizes the Oxford-IIIT Pets dataset, which is a widely used dataset for pet image segmentation tasks.
- Citation: [O. M. Parkhi et al., 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

## Acknowledgments

- The code includes dependencies and techniques from various open-source libraries and research papers.
- Special thanks to the TensorFlow team for providing deep learning frameworks and examples for semantic segmentation tasks.

## License

- The code in this repository is provided under the [MIT License](LICENSE).

## References

- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. In IEEE Conference on Computer Vision and Pattern Recognition (pp. 3498-3505).

- TensorFlow Examples Repository: [https://github.com/tensorflow/examples](https://github.com/tensorflow/examples)

- Oxford-IIIT Pets Dataset: [https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

---

