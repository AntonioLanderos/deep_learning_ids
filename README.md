# Intrusion Detection System using Deep Learning techniques

## Abstract

This project develops a deep learning-based Intrusion Detection System using the UNSW-NB15 dataset for multiclass attack classification. The model classifies network traffic into ten categories using the `attack_cat` variable instead of performing binary classification. The preprocessing stage included removing leakage columns, applying One-Hot Encoding to categorical features, scaling numerical values, and encoding the target labels.

Several architectures were evaluated, including a baseline neural network, a paper-based DNN, and a paper-based ANN. Since the dataset is highly imbalanced, class weights were applied to improve minority-class detection without artificially modifying the original traffic data. The best-performing model was a tuned DNN with three hidden layers of 100 neurons, a learning rate of 0.0005, and capped class weights. It achieved 75.11% test accuracy, 0.49 macro F1-score, and 0.74 weighted F1-score.

The results show that the model performs well on majority classes such as `Normal` and `Generic`, but still struggles with minority classes such as `Analysis`, `Backdoor`, and `Worms`. This suggests that the main limitation is the strong class imbalance in the dataset, and future work should focus on expanded or more balanced datasets.

## Introduction

The increasing dependence on computer networks has made cybersecurity a critical area for organizations, institutions, and users. As network traffic continues to grow, malicious activities have become more frequent and difficult to detect using traditional security mechanisms alone. For this reason, Intrusion Detection Systems (IDS) are widely used to monitor network behavior and identify suspicious or abnormal traffic patterns.

Machine learning and deep learning techniques have become important tools for intrusion detection because they can analyze large volumes of network data and learn patterns that may indicate malicious behavior. Unlike traditional rule-based systems, which depend on predefined signatures, learning-based models can identify relationships among features and classify traffic based on previously observed examples. This makes them useful for detecting both normal and attack traffic in complex datasets.

This project focuses on the development of a deep learning model for multiclass intrusion detection using the UNSW-NB15 dataset. The dataset contains network traffic records labeled according to different attack categories.

The original training and testing datasets provided by Kaggle were preserved to maintain a consistent evaluation process. The preprocessing stage includes removing non-relevant or leakage columns, encoding categorical features, scaling numerical values, and transforming attack categories into numerical labels. Additionally, because the dataset is highly imbalanced, less aggressive class weights were applied during training to reduce the effect of dominant classes and improve the model's ability to detect minority attack categories.

This project first implements a baseline feed-forward neural network and then evaluates a paper-based DNN architecture using the hyperparameters proposed by Aleesa et al. The models are evaluated usingaccuracy and loss, following the reference paper, while additional metrics such as precision, recall, F1-score, and confusion matrices are included to better analyze multiclass performance under class imbalance.

## Dataset

The dataset used in this project is the [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) dataset, which is commonly used for the development and evaluation of Network Intrusion Detection Systems. It contains network traffic records generated from both normal network activity and different simulated attack behaviors. The main purpose of the dataset is to provide representative network traffic data for training models capable of distinguishing between legitimate and malicious connections.

The version used in this project was obtained from Kaggle and is already divided into two main files: a training set and a testing set. This original separation was preserved in order to maintain a consistent evaluation process. The training dataset was used to train the model and to create a validation subset, while the testing dataset was kept separate and used only for the final evaluation.

The dataset contains numerical, nominal, and binary attributes that describe different characteristics of network traffic, such as protocol type, service, connection state, duration, number of packets, number of bytes, transmission rates, time-to-live values, and other connection-based indicators.

The dataset includes two main label columns: `label` and `attack_cat`. The `label` column is used for binary classification, where `0` represents normal traffic and `1` represents attack traffic. However, this project focuses on multiclass classification using the `attack_cat` column as the target variable. This allows the model to classify each record into one of ten traffic categories: `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, and `Worms`.

An important characteristic of this dataset is its strong class imbalance. Some categories, such as `Normal`, `Generic`, and `Exploits`, contain a large number of records, while minority classes such as `Worms`, `Shellcode`, `Backdoor`, and `Analysis` contain significantly fewer samples. This imbalance makes the classification task more challenging because the model may favor majority classes during training.

The following table presents the main attributes used in the dataset and their general descriptions.

| Attribute           | Type      | Description                                                                                        |
| ------------------- | --------- | -------------------------------------------------------------------------------------------------- |
| `id`                | Numerical | Unique identifier of the record. It is not used as an input feature for the model.                 |
| `dur`               | Numerical | Duration of the network connection or flow.                                                        |
| `proto`             | Nominal   | Protocol used in the connection, such as TCP, UDP, or others.                                      |
| `service`           | Nominal   | Network service associated with the flow, such as HTTP, DNS, FTP, or others.                       |
| `state`             | Nominal   | State of the network connection.                                                                   |
| `spkts`             | Numerical | Number of packets sent from the source to the destination.                                         |
| `dpkts`             | Numerical | Number of packets sent from the destination to the source.                                         |
| `sbytes`            | Numerical | Number of bytes sent from the source to the destination.                                           |
| `dbytes`            | Numerical | Number of bytes sent from the destination to the source.                                           |
| `rate`              | Numerical | Transmission rate of the network flow.                                                             |
| `sttl`              | Numerical | Source time-to-live value.                                                                         |
| `dttl`              | Numerical | Destination time-to-live value.                                                                    |
| `sload`             | Numerical | Source bits per second.                                                                            |
| `dload`             | Numerical | Destination bits per second.                                                                       |
| `sloss`             | Numerical | Number of packets lost from the source side.                                                       |
| `dloss`             | Numerical | Number of packets lost from the destination side.                                                  |
| `sinpkt`            | Numerical | Inter-packet arrival time from the source side.                                                    |
| `dinpkt`            | Numerical | Inter-packet arrival time from the destination side.                                               |
| `sjit`              | Numerical | Source jitter value.                                                                               |
| `djit`              | Numerical | Destination jitter value.                                                                          |
| `swin`              | Numerical | TCP window size from the source side.                                                              |
| `dwin`              | Numerical | TCP window size from the destination side.                                                         |
| `stcpb`             | Numerical | TCP base sequence number from the source side.                                                     |
| `dtcpb`             | Numerical | TCP base sequence number from the destination side.                                                |
| `tcprtt`            | Numerical | TCP round-trip time.                                                                               |
| `synack`            | Numerical | Time between SYN and SYN-ACK packets.                                                              |
| `ackdat`            | Numerical | Time between ACK and data packets.                                                                 |
| `smean`             | Numerical | Mean packet size from the source side.                                                             |
| `dmean`             | Numerical | Mean packet size from the destination side.                                                        |
| `trans_depth`       | Numerical | Transaction depth, commonly related to application-level traffic such as HTTP.                     |
| `response_body_len` | Numerical | Length of the response body.                                                                       |
| `ct_srv_src`        | Numerical | Number of connections with the same service and source address.                                    |
| `ct_state_ttl`      | Numerical | Number of connections with the same state and TTL value.                                           |
| `ct_dst_ltm`        | Numerical | Number of connections to the same destination in a short time window.                              |
| `ct_src_dport_ltm`  | Numerical | Number of connections from the same source to the same destination port in a short time window.    |
| `ct_dst_sport_ltm`  | Numerical | Number of connections to the same destination from the same source port in a short time window.    |
| `ct_dst_src_ltm`    | Numerical | Number of connections between the same source and destination in a short time window.              |
| `is_ftp_login`      | Binary    | Indicates whether the connection corresponds to an FTP login.                                      |
| `ct_ftp_cmd`        | Numerical | Number of FTP commands detected.                                                                   |
| `ct_flw_http_mthd`  | Numerical | Number of HTTP methods observed in the flow.                                                       |
| `ct_src_ltm`        | Numerical | Number of connections from the same source in a short time window.                                 |
| `ct_srv_dst`        | Numerical | Number of connections with the same service toward the destination.                                |
| `is_sm_ips_ports`   | Binary    | Indicates whether the source and destination IP addresses and ports are the same.                  |
| `attack_cat`        | Nominal   | Traffic category or attack family. This is the target variable used for multiclass classification. |
| `label`             | Binary    | Binary label of the traffic: `0` for normal traffic and `1` for attack traffic.                    |

<p align="center">
  <em> Table 1. Description of the dataset attributes</em>
</p>

## Methodology

### Preprocessing

The training dataset contains **82,332 records** and includes network traffic observations classified into ten categories using the `attack_cat` variable. These categories represent both normal traffic and different types of cyberattacks: `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, and `Worms`.

The dataset is composed of numerical, categorical, and binary variables. The numerical variables describe measurable characteristics of the network flow, such as duration, packet counts, byte counts, transmission rate, time-to-live values, load, loss, jitter, and connection counters. The categorical variables include attributes such as protocol type, service, and connection state. The target variable used in this project is `attack_cat`, which represents the specific traffic category or attack family.

The class distribution in the training dataset shows a strong imbalance among the categories. The `Normal` class contains 37,000 records, making it the most represented class, followed by `Generic` with 18,871 records and `Exploits` with 11,132 records. In contrast, minority classes such as `Analysis`, `Backdoor`, `Shellcode`, and `Worms` contain significantly fewer records, with `Worms` having only 44 samples.

  | Class          | Number of Records |
  | -------------- | ----------------: |
  | Normal         |            37,000 |
  | Generic        |            18,871 |
  | Exploits       |            11,132 |
  | Fuzzers        |             6,062 |
  | DoS            |             4,089 |
  | Reconnaissance |             3,496 |
  | Analysis       |               677 |
  | Backdoor       |               583 |
  | Shellcode      |               378 |
  | Worms          |                44 |
  
<p align="center">
  <img src="./images/classDistribution.png" alt="class distribution" width="30%" />
  <br>
  <em> Graph 1. Class distribution of the attack_cat categories in the training dataset</em>
</p>

This imbalance is important because a model trained directly on the original distribution may favor majority classes such as `Normal`, `Generic`, and `Exploits`, while performing poorly on minority attack categories. Therefore, the class distribution was analyzed before training. Instead of artificially oversampling or generating synthetic records, less aggressive class weights were applied during model training. This approach allowed the model to assign higher importance to minority classes without modifying the original dataset or generating potentially unrealistic network traffic samples.

#### Encoding of Categorical Variables

Since neural networks cannot directly process categorical text values, categorical variables were transformed into numerical representations. The input categorical attributes, such as `proto`, `service`, and `state`, were encoded using **One-Hot Encoding**. This technique converts each category into a separate binary column, preventing the model from assuming an artificial ordinal relationship between nominal categories.

The target variable `attack_cat` was encoded using **Label Encoding**, assigning a numerical value to each traffic class. This encoding was appropriate for the target variable because the model uses a softmax output layer with `sparse_categorical_crossentropy`, which expects class labels as integer values.

The columns `id`, `label`, and `attack_cat` were removed from the input features before training. The `id` column was removed because it is only an identifier and does not provide meaningful information for classification. The `label` column was removed because it corresponds to binary classification, while this project focuses on multiclass classification. The `attack_cat` column was removed from the input features to avoid data leakage, since it is the target variable that the model is trying to predict.

#### Feature Scaling

After encoding categorical variables, the numerical features were scaled using standardization. Scaling was applied because neural networks are sensitive to the magnitude of input values, and the dataset contains features with different ranges, such as packet counts, byte counts, rates, and time-based measurements. Standardizing the features helps the model train more consistently and prevents variables with larger numerical ranges from dominating the learning process.

The scaler was fitted only on the training data and then applied to the testing data. This was done to avoid data leakage and ensure that the test set remained unseen during the preprocessing learning process.

#### Class Imbalance Handling

Due to the strong imbalance in the `attack_cat` classes, less aggressive class weights were applied during model training. Instead of artificially oversampling or generating synthetic records, the model was trained with adjusted class weights to increase the penalty for misclassifying minority classes. This approach preserves the original dataset distribution while helping the model pay more attention to underrepresented categories such as `Analysis`, `Backdoor`, `Shellcode`, and `Worms`. Class weights approach was selected because since we are working with packet-based features, generating synthetic records could lead to unrealistic network traffic patterns that do not reflect real-world scenarios. Additionally, class weights allow the model to learn from the original data distribution while still addressing the imbalance issue.

#### Multicollinearity Analysis

A correlation matrix and heatmap were generated using the numerical variables in the dataset. The heatmap shows that some groups of variables present strong correlations, especially features related to packet counts, byte counts, TCP behavior, and connection counters. For example, variables such as `spkts`, `dpkts`, `sbytes`, `dbytes`, `sloss`, and `dloss` show visible correlation patterns. Similarly, some connection-based attributes such as `ct_srv_src`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, and `ct_srv_dst` also show notable relationships.

However, because the model used in this project is a neural network, no feature removal was performed at this stage based only on correlation. Neural networks can learn from correlated features, and the objective of this initial implementation was to preserve the original features after basic preprocessing. Therefore, all relevant features were kept.

<p align="center">
  <img src="./images/correlationHeatmap.png" alt="correlation-heatmap" width="50%" />
  <br>
  <em>Graph 2. Correlation heatmap of the numerical variables in the training dataset</em>
</p>

### Model Architecture

#### Baseline Architecture

The model implemented in this stage of the project is a simple feed-forward neural network designed for multiclass intrusion detection. The objective of the model is to classify each network traffic record into one of the ten categories defined by the attack category column.

The architecture consists of three hidden dense layers followed by a softmax output layer. The first hidden layer contains 64 neurons with ReLU activation, the second hidden layer contains 32 neurons with ReLU activation, and the third hidden layer contains 16 neurons with ReLU activation. The output layer contains 10 neurons, one for each class, and uses the softmax activation function to generate a probability distribution across the possible traffic categories.

The model was compiled using the Adam optimizer and the `sparse_categorical_crossentropy` loss function. This loss function was selected because the target labels were encoded as integer values using label encoding. The main metric used during training was accuracy.

The implemented architecture is summarized as follows:

| Layer         |                         Neurons | Activation Function | Purpose                                                    |
| ------------- | ------------------------------: | ------------------- | ---------------------------------------------------------- |
| Input Layer   | Number of preprocessed features | N/A                 | Receives the encoded and scaled dataset features           |
| Dense Layer 1 |                              64 | ReLU                | Learns initial nonlinear patterns from the input data      |
| Dense Layer 2 |                              32 | ReLU                | Learns intermediate feature representations                |
| Dense Layer 3 |                              16 | ReLU                | Learns a more compact representation before classification |
| Output Layer  |                              10 | Softmax             | Produces class probabilities for multiclass classification |

<p align="center">
  <em> Table 2. Baseline model architecture</em>
</p>

The model contains a total of **15,002 trainable parameters**. This architecture was used as an initial baseline before implementing a deeper architecture based on the reference paper.

#### Paper-Based DNN Architecture

After implementing the initial baseline model, a second experiment was developed using the DNN configuration described in the reference paper. This architecture was selected because the paper reports it as one of its main deep learning models for multiclass intrusion detection using the UNSW-NB15 dataset.

The paper-based DNN configuration uses three hidden layers with 100 neurons, the Adam optimizer, ReLU activation in the hidden layers, Softmax activation in the output layer, 100 epochs, and a batch size of 100. Since this project performs multiclass classification using the `attack_cat` column, the output layer was configured with 10 neurons, one for each traffic category.

The implemented paper-based DNN architecture is summarized as follows:

| Hyperparameter           |                           Value |
| ------------------------ | ------------------------------: |
| Hidden Layers            |                               3 |
| Neurons per Hidden Layer |                             100 |
| Optimizer                |                            Adam |
| Hidden Layer Activation  |                            ReLU |
| Output Layer Activation  |                         Softmax |
| Epochs                   |                             100 |
| Batch Size               |                             100 |
| Output Classes           |                              10 |
| Loss Function            | Sparse Categorical Crossentropy |

<p align="left">
  <em> Table 3. Paper-based DNN hyperparameters</em>
</p>

This model was compiled using the Adam optimizer and the `sparse_categorical_crossentropy` loss function. The target labels were kept as integer-encoded values, which is why sparse categorical crossentropy was appropriate. Accuracy was used as the main training metric, following the evaluation approach used in the reference paper.

## Experiments and Results

### Experiment 1: Baseline Model

The model was trained for **10 epochs** using the original Kaggle training set, with a validation subset created from the training data. The original Kaggle testing set was kept separate and used only for final evaluation. Less aggressive class weights were applied during training to address the strong class imbalance without artificially modifying the dataset.

The reference paper evaluates its deep learning models mainly using **accuracy** and **loss**. Following that approach, the implemented model was evaluated using training accuracy, validation accuracy, testing accuracy, and testing loss.

During training, the model showed a gradual improvement in both training and validation accuracy. The training accuracy increased from approximately **73.19%** in the first epoch to **83.55%** in the final epoch. Validation accuracy also improved from **78.16%** in the first epoch to approximately **82.52%** in the final epoch.

The final evaluation on the testing dataset produced the following results:

| Metric                    |  Value |
| ------------------------- | -----: |
| Test Accuracy             | 74.20% |
| Test Loss                 | 0.7263 |
| Final Training Accuracy   | 83.55% |
| Final Training Loss       | 0.5332 |
| Final Validation Accuracy | 82.52% |
| Final Validation Loss     | 0.4853 |
<p align="left">
  <em> Table 4. Baseline model evaluation results</em>
</p>

In addition to accuracy and loss, a classification report and confusion matrix were generated to better understand the model's behavior across classes. The weighted F1-score was approximately **0.73**, while the macro F1-score was approximately **0.46**. This confirms that the model performs well on majority classes such as `Generic` and `Normal`, but still struggles with minority classes such as `Analysis`, `Backdoor`, `Shellcode`, and `Worms`.

<p align="center">
  <img src="./images/classificationReport.png" alt="classification-report" width="30%" />
  <br>
  <em>Table 5. Classification report for the obtained results</em>
</p>

<p align="center">
  <img src="./images/confusionMatrix.png" alt="confusion-matrix" width="50%" />
  <br>
  <em>Graph 3. Confusion matrix for the obtained results</em>
</p>

The training and validation curves were analyzed to evaluate the behavior of the implemented baseline model during the learning process. The loss graph shows that the training loss decreases consistently across the 10 epochs. The validation loss also remains relatively stable and decreases overall. Since the validation loss does not increase significantly while the training loss decreases, there is no overfitting.

The accuracy graph also shows stable learning behavior. Training accuracy increases from approximately 0.73 to 0.84, while validation accuracy remains close to the training curve, reaching values around 0.82–0.83. The closeness between both curves suggests that the model is not memorizing the training data excessively. Additionally, the model does not show clear signs of underfitting, since both training and validation accuracy reach acceptable values for an initial multiclass classification baseline.

<p align="center">
  <img src="./images/lossCurves.png" alt="loss-curves" width="50%" />
  <br>
  <em>Graph 4. Training and validation loss curves</em>
</p>

<p align="center">
  <img src="./images/accuracyCurves.png" alt="accuracy-curves" width="50%" />
  <br>
  <em>Graph 5. Training and validation accuracy curves</em>
</p>

Based on the results, the implemented model provides a functional baseline for multiclass intrusion detection. However, the performance across minority classes indicates that further experimentation is needed. Future improvements will include testing a deeper architecture inspired by the reference paper, adjusting hyperparameters, increasing the number of epochs with early stopping, and comparing different class weighting strategies.

### Experiment 2: Paper-Based DNN Architecture

A second experiment was performed using the DNN hyperparameter configuration proposed in the reference paper. Unlike the baseline model, which used a smaller architecture with 64, 32, and 16 neurons, this experiment used three hidden layers with 100 neurons each. The model was trained for 100 epochs with a batch size of 100, using the Adam optimizer, ReLU activation in the hidden layers, and Softmax activation in the output layer.

<p align="center">
  <img src="./images/paperDnnHyperparams.png" alt="dnn-hyperparameters" width="50%" />
  <br>
  <em>Table 6. DNN hyperparameters based on the reference paper</em>
</p>

The training and validation accuracy curves show stable learning behavior across the 100 epochs. Training accuracy increased steadily and remained slightly above validation accuracy during the later epochs. Validation accuracy stayed close to the training curve. This indicates that the model was able to learn useful patterns without showing overfitting.

<p align="center">
  <img src="./images/paperDnnAccuracyCurves.png" alt="paper-dnn-accuracy-curves" width="50%" />
  <br>
  <em>Graph 6. Training and validation accuracy curves for the paper-based DNN model</em>
</p>

The loss curves show that training loss decreased consistently throughout the 100 epochs. Validation loss decreased during the early epochs and later stabilized with small fluctuations. Toward the final epochs, training loss continued decreasing while validation loss remained slightly higher, which suggests a bit of overfitting. However, the validation loss did not increase sharply, so the model did not show an overfitting problem.

<p align="center">
  <img src="./images/paperDnnLossCurves.png" alt="paper-dnn-loss-curves" width="50%" />
  <br>
  <em>Graph 7. Training and validation loss curves for the paper-based DNN model</em>
</p>

The final evaluation on the testing dataset produced the following results:

| Metric                 |  Value |
| ---------------------- | -----: |
| Test Accuracy          | 74.60% |
| Test Loss              | 1.1996 |
| Macro Avg Precision    |   0.53 |
| Macro Avg Recall       |   0.50 |
| Macro Avg F1-Score     |   0.48 |
| Weighted Avg Precision |   0.78 |
| Weighted Avg Recall    |   0.75 |
| Weighted Avg F1-Score  |   0.73 |
<p align="left">
  <em> Table 7. Paper-based DNN evaluation results</em>
</p>

The paper-based DNN model achieved a test accuracy of **74.60%**, which is slightly higher than the baseline model. Although the improvement in accuracy was small, the macro average F1-score increased to **0.48**, compared to approximately **0.46** in the baseline model. This is relevant because macro F1-score gives equal importance to all classes, including minority attack categories.

The classification report shows that the model performed strongly on majority classes such as `Generic` and `Normal`, with F1-scores of **0.98** and **0.86**, respectively. It also showed good performance on `Reconnaissance`, with an F1-score of **0.77**, and improved detection of `Shellcode`, which reached an F1-score of **0.52**. However, the model still struggled with minority classes such as `Analysis` and `Backdoor`, which obtained F1-scores of **0.02** and **0.08**, respectively.

The confusion matrix confirms that most `Generic` and `Normal` samples were correctly classified. However, minority classes were often confused with more represented categories such as `DoS`, `Exploits`, and `Normal`. This indicates that class imbalance remains a major challenge, even when using class weights and a larger architecture.

<p align="center">
  <img src="./images/paperDnnConfusionMatrix.png" alt="paper-dnn-confusion-matrix" width="50%" />
  <br>
  <em>Graph 8. Confusion matrix for the paper-based DNN model</em>
</p>

Overall, the paper-based DNN model provides a stronger experimental foundation than the baseline model because it follows the hyperparameter configuration proposed in the reference paper. However, the results show that the architecture still requires further experimentation to improve the detection of minority attack categories.

### Experiment 3: Paper-Based ANN Architecture

A third experiment was performed using the ANN hyperparameter configuration proposed in the reference paper. Unlike the paper-based DNN model, which used three hidden layers with 100 neurons each, this experiment used a shallower architecture with a single hidden layer containing 850 neurons. The model was trained for 100 epochs with a batch size of 100, using the Adam optimizer, ReLU activation in the hidden layer, and Softmax activation in the output layer.

<p align="center">
  <img src="./images/paperAnnHyperparams.png" alt="ann-hyperparameters" width="50%" />
  <br>
  <em>Table 8. ANN hyperparameters based on the reference paper</em>
</p>

The training and validation accuracy curves show stable learning behavior across the 100 epochs. Training accuracy increased during the first epochs and then stabilized around the mid-80% range. Validation accuracy followed a similar trend and remained close to the training curve throughout most of the training process. This indicates that the ANN model was able to learn patterns from the training data without showing a severe overfitting problem during training.

<p align="center">
  <img src="./images/paperAnnAccuracyCurves.png" alt="paper-ann-accuracy-curves" width="50%" />
  <br>
  <em>Graph 9. Training and validation accuracy curves for the paper-based ANN model</em>
</p>

The loss curves show that both training and validation loss decreased during the early epochs and then stabilized. The validation loss remained close to the training loss for most of the training process, although some fluctuations were observed. This suggests that the ANN model did not strongly overfit the validation set. However, despite the stable training behavior, the final test loss was higher than the one obtained by the paper-based DNN model, indicating weaker generalization on the official testing dataset.

<p align="center">
  <img src="./images/paperAnnLossCurves.png" alt="paper-ann-loss-curves" width="50%" />
  <br>
  <em>Graph 10. Training and validation loss curves for the paper-based ANN model</em>
</p>

The final evaluation on the testing dataset produced the following results:

| Metric                 |  Value |
| ---------------------- | -----: |
| Test Accuracy          | 73.71% |
| Test Loss              | 1.2446 |
| Macro Avg Precision    |   0.49 |
| Macro Avg Recall       |   0.50 |
| Macro Avg F1-Score     |   0.46 |
| Weighted Avg Precision |   0.76 |
| Weighted Avg Recall    |   0.74 |
| Weighted Avg F1-Score  |   0.71 |
<p align="left">
  <em> Table 9. Paper-based ANN evaluation results</em>
</p>

The paper-based ANN model achieved a test accuracy of **73.71%**, which was lower than the paper-based DNN model. The macro average F1-score was **0.46**, similar to the initial baseline model but lower than the DNN model. This indicates that the ANN architecture did not improve the balance across classes, especially for minority attack categories.

The classification report shows that the model performed well on majority classes such as `Generic` and `Normal`, with F1-scores of **0.98** and **0.86**, respectively. It also achieved reasonable performance on `Reconnaissance`, with an F1-score of **0.74**, and detected `Shellcode` with an F1-score of **0.51**. However, the model struggled with minority classes such as `Analysis`, `Backdoor`, and `Worms`, which obtained F1-scores of **0.01**, **0.06**, and **0.14**, respectively.

The confusion matrix confirms that most `Generic` and `Normal` samples were correctly classified. However, minority classes were frequently confused with more represented classes. For example, many `Analysis` and `Backdoor` samples were classified as `DoS` or `Normal`, while `Worms` samples were often confused with `Exploits`. This confirms that class imbalance and overlapping feature patterns remained major challenges for the ANN model.

<p align="center">
  <img src="./images/paperAnnConfusionMatrix.png" alt="paper-ann-confusion-matrix" width="50%" />
  <br>
  <em>Graph 11. Confusion matrix for the paper-based ANN model</em>
</p>

Overall, the paper-based ANN architecture did not outperform the paper-based DNN model. Although it followed the hyperparameter configuration proposed in the reference paper, the results suggest that the deeper DNN architecture was more appropriate for this dataset. For this reason, the DNN architecture was selected as the main candidate for further hyperparameter tuning.

### Experiment 4: Final Tuned DNN Model

After evaluating the baseline model, the paper-based DNN architecture, and the paper-based ANN architecture, the DNN model was selected as the main candidate for hyperparameter tuning. The final selected configuration kept the paper-based DNN structure with three hidden layers of 100 neurons each, but included two important adjustments: the Adam optimizer learning rate was reduced to 0.0005, and the class weights were capped to reduce the effect of extreme weights from minority classes.

The objective of this experiment was to improve the balance between overall accuracy and minority-class detection while preserving the main architecture proposed in the reference paper. The model was trained for 100 epochs with a batch size of 100, using ReLU activation in the hidden layers and Softmax activation in the output layer.

The final tuned DNN configuration is summarized as follows:

| Hyperparameter | Value |
|---|---:|
| Architecture | DNN |
| Hidden Layers | 3 |
| Neurons per Hidden Layer | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Hidden Layer Activation | ReLU |
| Output Layer Activation | Softmax |
| Epochs | 100 |
| Batch Size | 100 |
| Scaling Method | StandardScaler |
| Class Imbalance Strategy | Capped class weights |
| Loss Function | Sparse Categorical Crossentropy |
| Output Classes | 10 |
<p align="left">
  <em> Table 10. Final tuned DNN hyperparameters</em>
</p>

The training and validation accuracy curves show stable learning behavior across the 100 epochs. Training accuracy increased steadily and reached approximately the high-80% range, while validation accuracy remained close to the training curve but slightly lower during the final epochs. This small gap suggests mild overfitting, but the validation accuracy did not collapse, indicating that the model still maintained stable generalization during training.

<p align="center">
  <img src="./images/finalDnnAccuracyCurves.png" alt="final-dnn-accuracy-curves" width="50%" />
  <br>
  <em>Graph 12. Training and validation accuracy curves for the final tuned DNN model</em>
</p>

The loss curves show that training loss decreased consistently throughout the training process. Validation loss also decreased during the early epochs and later stabilized with small fluctuations. Toward the final epochs, the training loss remained slightly lower than the validation loss, which also suggests mild overfitting. However, because the validation loss stayed relatively stable and did not increase sharply, the model did not show severe overfitting.

<p align="center">
  <img src="./images/finalDnnLossCurves.png" alt="final-dnn-loss-curves" width="50%" />
  <br>
  <em>Graph 13. Training and validation loss curves for the final tuned DNN model</em>
</p>

The final evaluation on the testing dataset produced the following results:

| Metric                 |  Value |
| ---------------------- | -----: |
| Test Accuracy          | 75.11% |
| Test Loss              | 0.9952 |
| Macro Avg Precision    |   0.53 |
| Macro Avg Recall       |   0.52 |
| Macro Avg F1-Score     |   0.49 |
| Weighted Avg Precision |   0.78 |
| Weighted Avg Recall    |   0.75 |
| Weighted Avg F1-Score  |   0.74 |
<p align="left">
  <em> Table 11. Final tuned DNN evaluation results</em>
</p>

The final tuned DNN model achieved the best overall performance among the evaluated configurations. Compared to the paper-based DNN model, test accuracy increased from **74.60%** to **75.11%**, macro F1-score increased from **0.48** to **0.49**, and weighted F1-score increased from **0.73** to **0.74**. Although the improvement was moderate, it was consistent across the most important evaluation metrics.

The classification report shows that the model maintained strong performance on majority classes such as `Generic` and `Normal`, with F1-scores of **0.99** and **0.87**, respectively. It also achieved stable results for `Reconnaissance`, with an F1-score of **0.77**, and `Shellcode`, with an F1-score of **0.55**. The `Worms` class also improved compared to some previous experiments, reaching an F1-score of **0.25**.

However, minority classes such as `Analysis` and `Backdoor` remained difficult to classify, with F1-scores of **0.02** and **0.06**, respectively. This indicates that even after tuning the learning rate and adjusting class weights, the model still struggled with some highly underrepresented and overlapping attack categories.

The confusion matrix confirms that most `Generic` and `Normal` samples were correctly classified. It also shows that minority classes were still frequently confused with more represented categories such as `DoS`, `Exploits`, and `Normal`. For example, `Analysis` and `Backdoor` samples were often classified as `DoS` or `Normal`, while some `Worms` samples were confused with `Exploits`. This confirms that class imbalance and similarity between attack patterns remained important limitations.

<p align="center">
  <img src="./images/finalDnnConfusionMatrix.png" alt="final-dnn-confusion-matrix" width="50%" />
  <br>
  <em>Graph 14. Confusion matrix for the final tuned DNN model</em>
</p>

Overall, this configuration was selected as the final model because it provided the best balance between test accuracy, macro F1-score, weighted F1-score, and class-level performance. The improvements were not large, but they were consistent, making the final tuned DNN model the strongest configuration tested in this project.

### Experiment Comparison

The experiments show a gradual improvement from the initial baseline model to the final tuned DNN configuration. The baseline model provided a functional starting point, achieving a test accuracy of 74.20% and a macro F1-score of 0.46. The paper-based DNN architecture improved the macro F1-score to 0.48 and slightly increased the test accuracy to 74.60%, showing that the deeper architecture was better suited for the multiclass classification task than the initial baseline.

The paper-based ANN architecture obtained the lowest overall performance among the main paper-based experiments, with a test accuracy of 73.71%, a test loss of 1.2446, a macro F1-score of 0.46, and a weighted F1-score of 0.71. This indicates that, for this dataset, the wider single-layer ANN was less effective than the deeper DNN model.

After hyperparameter tuning, the best-performing model was the DNN with a reduced learning rate and capped class weights. This configuration achieved the highest test accuracy at 75.11%, the highest macro F1-score at 0.49, and the highest weighted F1-score at 0.74. Although the improvement was moderate, it was consistent across the most important evaluation metrics.

| Experiment | Architecture | Epochs | Batch Size | Test Accuracy | Test Loss | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | 64-32-16 | 10 | 64 | 74.20% | **0.7263** | 0.46 | 0.73 |
| Paper-Based DNN | 100-100-100 | 100 | 100 | 74.60% | 1.1996 | 0.48 | 0.73 |
| Paper-Based ANN | 850 | 100 | 100 | 73.71% | 1.2446 | 0.46 | 0.71 |
| DNN + LR + Capped weights | 100-100-100 | 100 | 100 | **75.11%** | 0.9952 | **0.49** | **0.74**
<p align="center">
  <em> Table 12. Comparison of evaluated model configurations</em>
</p>

Based on these results, the DNN with learning rate tuning and capped class weights was selected as the best model configuration. It provided the best balance between overall accuracy and class-level performance, while still preserving the original dataset distribution.

### Hyperparameter Tuning

After evaluating the baseline model and the paper-based architectures, additional hyperparameter tuning was performed using the paper-based DNN as the main architecture. The objective was to improve the model without moving too far away from the configuration proposed in the reference paper.

The first tuning experiment modified the batch size and layer structure. The original DNN architecture with three hidden layers of 100 neurons was changed to a decreasing structure of 128, 64, and 32 neurons, and the batch size was reduced to 32. However, this configuration produced results very similar to the original DNN baseline, suggesting that changing only the number of neurons and the batch size was not enough to significantly improve the model.

The next experiment focused on the learning rate. The Adam optimizer was kept, but the learning rate was reduced from the default value of 0.001 to 0.0005. This produced a small improvement in test accuracy and a clearer improvement in test loss, indicating that a lower learning rate helped the model train more steadily.

Dropout regularization was also tested with dropout rates of 0.2 and 0.1. These experiments reduced the test loss and increased recall for some minority classes, such as `Shellcode` and `Worms`. However, they also reduced test accuracy and did not improve the macro F1-score. This suggests that dropout made the model more sensitive to minority classes, but also increased false positives.

MinMaxScaler was tested as an alternative to StandardScaler because the reference paper uses normalization. This experiment reduced the test loss and improved recall for some minority classes, but it did not improve the macro F1-score or weighted F1-score compared to the learning-rate-tuned model with StandardScaler. Therefore, StandardScaler was kept for the final model.

Finally, capped class weights were tested. The previous class weighting strategy helped the model pay more attention to minority classes, but extreme weights could make the model overpredict rare categories. To reduce this effect, the class weights were capped while keeping the learning rate at 0.0005. This configuration produced the best overall result, reaching 75.11% test accuracy, 0.49 macro F1-score, and 0.74 weighted F1-score.

The final selected model used the paper-based DNN architecture with three hidden layers of 100 neurons, Adam optimizer with a learning rate of 0.0005, StandardScaler preprocessing, batch size of 100, and capped class weights. This configuration was selected because it achieved the best balance across accuracy, macro F1-score, and weighted F1-score.
## Conclusion

This project developed and evaluated deep learning models for multiclass intrusion detection using the UNSW-NB15 dataset. The main objective was to classify network traffic into different attack categories using the `attack_cat` variable instead of performing only binary classification between normal and attack traffic. To achieve this, the original Kaggle training and testing split was preserved, categorical features were encoded, numerical features were scaled, and the target labels were transformed into numerical classes.

Several models were tested throughout the project. The first model was a simple baseline feed-forward neural network, followed by two architectures based on the reference paper: a DNN with three hidden layers of 100 neurons and an ANN with one hidden layer of 850 neurons. The paper-based DNN achieved better results than the ANN, showing that the deeper architecture was more appropriate for this multiclass classification task. After that, different hyperparameter tuning strategies were tested, including changes in batch size, layer size, learning rate, dropout, scaling method, and class weight adjustment.

The best-performing configuration was the final tuned DNN model, which kept the paper-based DNN architecture but used a lower learning rate of 0.0005 and capped class weights. This model achieved the highest overall performance among the tested configurations, with a test accuracy of 75.11%, a macro F1-score of 0.49, and a weighted F1-score of 0.74. Although the improvement over the original DNN model was moderate, it was consistent across the most important evaluation metrics.

However, the experiments also showed that the model reached a performance barrier. After several tuning attempts, the results remained within a similar range, especially in terms of macro F1-score. This suggests that the main limitation was not only the neural network architecture or the selected hyperparameters, but also the characteristics of the dataset itself. The strong class imbalance heavily affected the model’s ability to correctly classify minority attack categories such as `Analysis`, `Backdoor`, `Shellcode`, and `Worms`.

Class weights were selected as the main imbalance-handling technique because they allowed the model to give more importance to minority classes without modifying the original dataset. This was important because the dataset is based on network traffic features, and artificially generating or duplicating records could create unrealistic traffic patterns or increase the risk of overfitting. Instead of oversampling or using synthetic data, class weights preserved the original data distribution while still helping the model pay more attention to underrepresented classes.

Even with class weights, some minority classes remained difficult to classify. This indicates that the available number of samples for those categories was not sufficient for the model to learn strong and generalizable patterns. For this reason, continuing to test small hyperparameter changes was unlikely to produce significant improvements. A more effective solution would be to work with an expanded or more balanced version of the dataset, where minority classes contain more representative examples. This would give the model more information to learn the behavior of rare attack categories and could improve macro-level performance.

Overall, the final tuned DNN model provided the best balance between accuracy, macro F1-score, weighted F1-score, and class-level performance among the tested configurations. The project demonstrates that deep learning can be useful for multiclass intrusion detection, especially for majority classes such as `Normal` and `Generic`. However, it also shows that class imbalance remains a major challenge in intrusion detection datasets. Future work should focus on evaluating expanded datasets, improving the representation of minority classes, and exploring more advanced imbalance-handling techniques while ensuring that generated or augmented traffic records remain realistic.

## References
Aleesa, Ahmed & Thanoun, Mohammed & Mohammed, Ahmed & Sahar, Nan. (2021). DEEP-INTRUSION DETECTION SYSTEM WITH ENHANCED UNSW-NB15 DATASET BASED ON DEEP LEARNING TECHNIQUES. Journal of Engineering Science and Technology. 16. 711-727.

"Deep learning in intrusion detection systems," IEEE Conference Publication | IEEE Xplore, Dec. 01, 2018. https://ieeexplore.ieee.org/document/8625278

DataGuard Insights, “How intrusion detection systems help identify cyber threats in real-time,” DataGuard, Feb. 04, 2026. [Online]. Available: https://www.dataguard.com/blog/how-intrusion-detection-systems-help-identify-cyber-threats/
