## Preprocessing The Data 
For preprocessing I used NLTK (Natural Language Toolkit) library in Python, with this 
library I have made following operations 
- removal of stopwords

   Stopwords are common words in a language (such as "and," "the," "is," etc.) that are often 
filtered out before processing natural language data. These words are typically removed in 
text preprocessing to reduce noise and focus on the more informative words. 
- removal of punctuations
 
   Punctuation can often be considered noise, so removing it can help clean the text for further 
analysis. 
- lemmatizing
   
   Lemmatization is the process of reducing words to their base or root form, known as the 
lemma. It is a crucial step in natural language processing (NLP) for normalizing text, 
improving text matching, and reducing dimensionality. 
- tokenization 

   Tokenization is the process of splitting text into smaller units, such as words or sentences. 
These units, called tokens, are essential for various natural language processing (NLP) tasks.
 
## Model Architecture 
The model starts with an embedding layer that converts each word in the input sequences 
into a 5-dimensional vector.

These embeddings are then fed into a SimpleRNN layer with 32 units, which processes the 
sequences and outputs the final hidden state. 

Finally, the output from the RNN layer is passed through a dense layer with 3 units and a 
softmax activation function to produce a probability distribution over the 3 classes. 

## Training and Evaluation 
The optimizer is responsible for updating the model's weights during training. The Adam (Adaptive 
Moment Estimation) optimizer is used in our model. Adam is popular because it adjusts the learning 
rate based on the first and second moments of the gradients, providing faster convergence and better 
performance. 

The loss function used for model is ‘Categorical Cross Entropy’. It is used for multi-class 
classification problems where each sample belongs to exactly one of the categories. It calculates the 
difference between the predicted probability distribution and the true distribution (one-hot encoded 
labels). Minimizing this loss helps improve the model's accuracy for multi-class classification tasks. 

Final model achieved an accuracy of %71 on the test dataset. These metrics indicate that the 
model performs well in distinguishing between different sentiments, although there is room for 
improvement. 
