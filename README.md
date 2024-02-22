## CSE151A-Project-Repo
Machine Learning Project in Python

## Jupyter Notebook
<a target="_blank" href="https://colab.research.google.com/github/kaijia2022/CSE151A-Project-Repo/blob/main/StarType_Prediction.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Preprocessing Plan MS1-2
We plan on viewing our data and looking at any categorical values and one hot encoding them. We see that there is a star color and a spectral class, so we will one hot encode those. They are categorical data that are nominal and we do not want to have one to be greater than the other. We will then use min max scaling to normalize each field of the input data. 

## Data Preprocessing MS3 Answers
4. **Where does your model fit in the fitting graph.**
   The model appears to fit in the ideal range in the fitting graph. With the MSEs, the model seems to be making the correct predictions and not overfitting or underfitting. The test data MSE is just above the training data. 

5. **What are the next 2 models you are thinking of and why?**
We are thinking of doing logistic regression and a neural network. We think that logistic regression is better for classification of multiple classes, and that neural networks would be more accurate and can predict more types.

7. **Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?**
Our 1st model concludes that linear regression looks like a good fit for the data, but it does not consider that this is a classification problem. The MSE only indicates the continuous values, rather than the probability of what star type it may be. To improve it, models like logistic regression and a neural network may need to be used. These models are better suited for classification problems than linear regression is. If we strictly wanted to improve the linear regression model, we would need to collect more data and select better correlated features.

## Our Contributors :

<a href="https://github.com/kaijia2022/CSE151A-Project-Repo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kaijia2022/CSE151A-Project-Repo" />
</a>
<br>
<br>
