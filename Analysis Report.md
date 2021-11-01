# Deep Learning Homework: Charity Funding Predictor

1. **Overview** 
    The purpose of the analysis is to create a machine learning model using neural networks that can predict whether Alphabet Soup funding applicants will be successful. The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. The analysis was performed by using a dataset of Alphabet Soup applicants that have received funding and using the metadata to create features to train a model. Included in the analysis are multiple changes to the features and model in an attempt to optimize the result.

2. **Results**: 

  * Data Preprocessing
    * What variable(s) are considered the target(s) for the model?
      The model uses the "IS_SUCCESSFUL" column from the data set as the target for training and testing the model.
    * What variable(s) are considered to be the features for the model?
      The features include the following from the data set:
      - **APPLICATION_TYPE** — Alphabet Soup application type. Two different binning approaches were used to group application types with low value counts into an "other" category.
      - **AFFILIATION** — Affiliated sector of industry
      - **CLASSIFICATION** — Government organization classification. Two different binning approaches were used to group classifications with low value counts into an "other" category.
      - **USE_CASE** — Use case for funding
      - **ORGANIZATION** — Organization type
      - **STATUS** — Active status
      - **INCOME_AMT** — Income classification
      - **SPECIAL_CONSIDERATIONS** — Special consideration for application
      - **ASK_AMT** — Funding amount requested
    * What variable(s) are neither targets nor features, and should be removed from the input data?
      The variables that were removed from the input data include:
      **EIN** and **NAME**—Identification columns
    * Get_dummies was used to turn categorical data to numeric and then the data set was split into training and test sets and  scaled. The data preprocessing steps were performed using Jupyter Lab and the notebooks were uploaded to Colab to develop the models.
    * Steps for Compiling, Training, and Evaluating the Model
      The following steps were taken to develop and evaluate the model:
      - The model was defined using tf.keras.models.Sequential.
      - The hidden layers and output layer with activitation methods were defined
      - The model was then compiled with the "adam" optimizer
      - Finally, the model was fit and epoch parameter set.
    * How many neurons, layers, and activation functions were selected for the neural network model, and why?
      The original model included the following layers:
      - 1 hidden layer (input) with 21 neurons and the relu activitation function
      - 2nd hidden layer with 21 neurons and the relu activation function
      - Output layer with 1 neuron and the sigmoid activation function.
      These parameters were chosen based on the number of features. There does not appear to be a "rule of thumb" based on the number of features, but Google research indicates that neurons should be set to 1/2 the number of features and the more features the highet the number of epochs. Therefore, with 43 features, I set the number of neurons to 21 and the number of epochs to 90.
    * Did the model achieve the target model performance?
      Unfortunately, the model and attempts at optimization were not able to achieve the target performance of 75%. 
    * What steps were taken to try and increase model performance?
      To improve the performance I made the following modifications:
      - Changed the number a neurons, activation functions, and epochs in various combinations
      - Changed the features by changing the binning on the application_type and classification features to increase the features from 43 to 51.
      - I created a total of 4 additional models in an attempt to improve on the original results. Two of the optimized models were based on the original features and 2 optimization models were based on the revised 51 features.

3. **Summary**: 
    Unfortunately, I was not able to create a model that met the target performance. The best result I was able to attain was and accuracy of 73.02% and the loss on all models was over 50%. It appears that the models with fewer neurons perform slightly better, but not by any significance. I think the performance might be able to be improved by reducing the number of features by eliminated data that may not be relevant. For example, affiliation and status may not have any significance for the outcome. By reducing the number of features and the number of neurons the performance may be improved. The other option would be to use a supervised learning model, such as Random Forest, to see if performance can be improved.
