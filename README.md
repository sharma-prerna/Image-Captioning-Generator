# Image-Captioning-Generator
**Description**: Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English. It takes an image as an input and generates the most relevant caption for that image.

---

**Model Architecture**: Since the input consists of two parts, an image vector and a partial caption, we cannot use the Sequential API provided by the Keras library. For this
reason, we use the Functional API which allows us to Merge Models. First, let’s look at the brief architecture which contains the high level sub-modules:
![High-level-Architecture](https://github.com/sharma-prerna/Image-Captioning-Generator/blob/main/ICG-architecture.png)

---

**Dataset Used** : There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But for the purpose of this case study, I have used the Flickr 8k dataset which you can download from this link: https://drive.google.com/drive/folders/1j-LPG-gnziaKQGajOo9L-99rKC3do9zo?usp=sharing Also training a model with large number of images may not be feasible on a system which is not a very high end PC/Laptop. This dataset contains 8000 images each with 5 captions (as we have already seen in the Introduction section that an image can have multiple captions, all being relevant simultaneously). These images are bifurcated as follows:
1. Training Set — 6000 images
2. Test Set — 2000 images

---

**Technologies Used**
|Technology|Property|
|---|---|
|Python|Programming Language|
|CNN|Image processing|
|LSTM|Caption processing|
|Google Colab|Platform to run the model|


