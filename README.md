
## Dimitry Kirtsman Data Science Projects

### Enhancing LLMs for Domain-Specific Tasks Through Dataset Creation, Model Fine-tuning, and Retrieval-Augmented Generation(RAG)
The analysis presents a series of experiments to enhance the performance of a Llama 2-7B model in generating accurate answers for a specific domain, using an HP computer troubleshooting guide as a case study. The analysis employs a multi-pronged approach, starting with the generation of a question-answer pair dataset from the troubleshooting guide using ChatGPT (GPT-4). Next,  a Llama 2-7B model is fine-tuned using the dataset. Lastly, retrieval-augmented generation (RAG) techniques with Pinecone are explored for improved performance. A detailed account of each experiment, the methodologies employed, and the outcomes are provided, offering insights into the challenges and potential solutions of using generative AI in this context.

**Software:** Python

**Methods:** Llama 2, ChatGPT API, LangChain, LLM Fine-tuning, Pinecone and Retrieval-Augmented Generation.

[Link to Analysis Discussion](https://drive.google.com/file/d/16UefmGMvpfr_1q9l21pN6KDDB0b-mqyx/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Llama-2-Fine-Tuning-and-RAG-Models-)  

<br />

### Forecasting Stock Prices Using LSTM and Transformer Architectures
The paper investigated the efficacy of LSTM and Transformer models in forecasting one-day ahead stock price returns. In addition, both models used the Time2Vec algorithm as a positional embedding layer. The stocks considered in the analysis were Amazon (AMZN), Intel (INTC), Pfizer (PFE), Proctor & Gamble (PG), and Verizon (VZ). The models used a ten-year time series from October 1, 2012, to September 30, 2022, with price, volume, and a variety of technical indicators serving as the features for each model. Training for all models used a cross-validation design with hyperparameter tuning. RMSE and MAPE were utilized to evaluate price level forecast performance. In addition, binary (up/down) signals were created from the price forecasts and assessed using accuracy and F1 scores. Study results showed no significant performance differences between the two models.

**Software:** Python, Keras/TensorFlow

**Methods:** LSTMs, Transformers, Time2Vec, Time series, Cross-validation and Hyperparameter Tunning. 

[Link to Paper](https://drive.google.com/file/d/1AqRlX8aUwSOF8vcj7Sj1nF6uQ17JUnL0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Stock_Models/tree/main/Model%20Code)  

<br />


### Classification of Cyberbullying Tweets Using LSTM, GRU, CNN, and Transformer Models
The paper explores the efficacy of LSTM, GRU, CNN, and Transformer models for text classification of cyberbullying tweets. The analysis consists of 8 experiments (12 models in all) and utilizes a dataset of more than 47,000 cyberbullying tweets and the Keras library in Python. The experiments start with data cleaning and EDA. The next five experiments utilize cross-validation and hyperparameter tuning to test various structures of LSTMs, GRUs, their bidirectional counterparts, and CNNs. These experiments utilize trained embedding and pre-trained tweet-based GloVe embeddings. The last two experiments involve an attention-based transformer and a fine-tuned DistilBERT model. The best model from the analysis is the DistilBERT model, which obtained an 89.44% accuracy on the test set.

**Software:** Python, Keras/TensorFlow

**Methods:** LSTMs, GRUs, CNNs, Transformers, GloVe Embeddings, Cross-validation and Hyperparameter Tuning.

[Link to Paper](https://drive.google.com/file/d/1pVNc4LXxP6sw9A7DGvD6xWizAn3vEtdq/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Cyberbullying-Tweets-Models/tree/main/Model_Code)  

<br />

### Image Classification Using ANN, CNN, and ResNet50 Neural Network Architectures
The paper explores the selection of a proper deep neural network architecture, ANN, CNN, or the ResNet50 transfer learning model for image classification. The analysis consists of 7 experiments (12 models in all), utilizing the CIFAR-10 dataset and the Keras library in Python. The experiments start with the analysis of ANNs and CNNs with two and then three hidden layers (convolution/max-pooling layers for CNNs). Next, regularization is introduced for all of the models in the form of dropout and batch normalization. Additional experiments are conducted utilizing hyperparameter tuning to improve performance and data augmentation to further decrease overfitting. The final experiment explores transfer learning using the ResNet50 CNN architecture, which achieved the best CIFAR-10 test set accuracy score of all the models at 86.18%.

**Software:** Python, Keras/TensorFlow

**Methods:** ANNs, CNNs, ResNet50, Transfer Learning, Data Augmentation, Cross-validation and Hyperparameter Tuning. 

[Link to Paper](https://drive.google.com/file/d/1LUuux5frpF5OSHiTokXstBN2hpwAPiC0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Image-Classification-Models/tree/main/Model%20Code) 

<br />

### DBSCAN, Isolation Forest, and Autoencoders for Anomaly Detection of Sales Transaction Data
The paper explores the efficacy of DBSCAN, Isolation Forest, and Autoencoders in detecting fraudulent sales transactions. The dataset contains variables related to product sales and is comprised of 133,731 train set observations and 15,732 test set observations. Data cleaning and feature creation are first applied to both data sets. Each of the models is then fit on the train set, and predictions of fraud/not fraud are obtained from the test set. Different iterations for each of the models above are tested by varying model parameters. Model performance is then evaluated by comparing the model’s test set predictions of fraud/not fraud versus the actual fraud outcomes using the F1 score.

**Software:** R

**Methods:** DBSCAN, Isolation Forest, Autoencoders, Anomaly Detection. 

[Link to Paper](https://drive.google.com/file/d/1fBd5rWaBaUj7aAlIoP2GofhubaVUnser/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Anomaly-Detection-Models/tree/main/Model%20Code) 

<br />


### Estimating the Number of Medical Office Visits with Count Data Models
Count data models are used for dependent variables that involve counts, such as the number of children in a family or the number of international trips a person takes per year. The current analysis applied a variety of count data models to estimate the number of physician office visits by the elderly. The five models used in the analysis were the Poisson regression, the Poisson regression with dispersion, the Negative Binomial regression, the Hurdle regression, and the Zero-Inflated regression. Cross-validation design was utilized to evaluate model performance using a variety of metrics such as BIC, AIC, MSE, and MAE. Patient segment classification analysis was also conducted to determine if the models reached specific business metrics in order to be put into production.

**Software:** R

**Methods:** Poisson regression, Poisson regression with dispersion, Negative Binomial regression, Hurdle regression, Zero-Inflated regression, Cross-validation. 

[Link to Paper](https://drive.google.com/file/d/15cBKUZJbgpW1qtGvMDLm3GVPPXkKLcBj/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Count-Data-Models/tree/main/Model%20Code) 

<br />


### Classification with Random Forests, Extra Trees, XGBoost, and Stacked Models
The analysis applied Random Forests, Extra Trees, XGBoost, and stacked models for classification. Exploratory data analysis was conducted with a variety of new variables created, including one-hot encoded and target-encoded features. The analysis utilized a cross-validation design with hyperparameter tuning. Pipelines were created for both the preprocessing and modeling steps before being utilized by the GridSearchCV algorithm. The models were used to predict survival with data from the Titanic dataset. Unlike the research above, the analysis does not rise to the level of a full-fledged paper. However, it does highlight the ability to implement all of the aforementioned machine learning tools.

**Software:** Python

**Methods:** Random Forests, Extra Trees, XGBoost, Stacked Model, Target Encoding, Cross-validation, and Hyperparameter Tuning. 

[Link to Paper](https://drive.google.com/file/d/1eukKiXnb2wHiAqozQeN5-Ue2kuTnjK7p/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Ensemble_Models/tree/main/Model%20Code)
