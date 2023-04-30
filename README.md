Download Link: https://assignmentchef.com/product/solved-fman45-assignment-2-solving-a-nonlinear-kernel-svm-with-hard-constraints
<br>
Solving a nonlinear kernel SVM with hard constraints.

In the lecture we have seen the linear Support Vector Machine (SVM) for binary classification, which aims to find parameters <strong>w </strong>∈ R<em><sup>d </sup></em>and a bias term <em>b </em>∈ R such that <strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub></em>+<em>b </em>≥ 1 if the example <strong>x</strong><em><sub>i </sub></em>∈ R<em><sup>d </sup></em>belongs to the positive class (<em>y<sub>i </sub></em>= +1), and <strong>wx</strong><em><sub>i </sub></em>+ <em>b </em>≤ −1 when <strong>x</strong><em><sub>i </sub></em>belongs to the negative class (<em>y<sub>i </sub></em>= −1). The size of the margin is inverse proportional to the length of ||<strong>w</strong>||. The margin should be as large as possible, which corresponds to minimizing ||<strong>w</strong>||, or rather  for computational reasons. In this section and the next, we will consider the theoretical aspects of SVM classifiers. First, we examine the linear hard margin SVM, which is defined as the solution to the minimization problem:

minimize(1)

<strong>w</strong><em>,b</em>

subject to         <em>y<sub>i</sub></em>(<strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub></em>+ <em>b</em>) ≥ 1<em>, </em>∀<em>i</em>

Consider the one-dimensional binary classification problem with classes ‘+1’ and ‘-1’:

(2)

As can be seen, the dataset is not linearly separable. Instead of trying to solve the above SVM problem, we therefore consider the (non-linear) feature map

(3)

and the corresponding kernel is given by <em>k</em>(<em>x,y</em>) = <em>φ</em>(<em>x</em>)<sup>&gt;</sup><em>φ</em>(<em>y</em>) and the kernel matrix by

<table width="571">

 <tbody>

  <tr>

   <td width="553"><strong>K </strong>= [<em>k</em>(<em>x</em><em>i</em><em>,x</em><em>j</em>)]1≤<em>i,j</em>≤4Task T1: (5 p) Compute the kernel matrix <strong>K </strong>using the data from the table.The Lagrangian dual problem for the (hard margin) SVM is given by:4                                 4</td>

   <td width="18">(4)</td>

  </tr>

 </tbody>

</table>

(5)

=1

subject to            <em>α<sub>i </sub></em>≥ 0 and <sup>X</sup><em>y<sub>i</sub>α<sub>i </sub></em>= 0<em>,             </em>∀<em>i</em>

<em>i</em>=1

Task T2: (5 p) Solve the maximization problem in (5) for <em>α </em>numerically, using the data in (2). You may use, without a proof that, for the data in (2), the solution satisfies <em>α </em>=

<em>α</em><sub>1 </sub>= <em>α</em><sub>2 </sub>= <em>α</em><sub>3 </sub>= <em>α</em><sub>4</sub>.

Next, we know that, for any support vector <em>x<sub>s</sub></em>, we have

(6)

and that the equation for the classifier is given by

(7)

Task T3: (10 p) For the data-target pairs in (2), reduce the classifier function (7) to the most simple the simplest possible form, leading to a simple polynomial in <em>x</em>.

Now consider another binary classification problem with classes ‘+1’ and ‘-1’, and data:

(8)

Task T4: (5 p) With the same kernel <em>k</em>(<em>x,y</em>) as above, what is the solution <em>g</em>(<em>x</em>) of the nonlinear kernel SVM with hard constraint on the dataset in (8)? Explain how you got to this solution.

<h1>2           The Lagrangian dual of the soft margin SVM</h1>

In the soft-margin SVM, we allow errors <em>ξ<sub>i </sub></em>in this classification tasks for each data point <strong>x</strong><em><sub>i</sub></em>. We denote the collection of all <em>ξ<sub>i </sub></em>by <em>ξ</em>. The primal formulation of the linear soft margin classifier is given by

minimize(9)

<strong>w</strong><em>,b,</em><em>ξ</em>

subject to         <em>y<sub>i</sub></em>(<strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub></em>+ <em>b</em>) ≥ 1 − <em>ξ<sub>i</sub></em>

<em>ξ<sub>i </sub></em>≥ 0

Task T5: (10 p) Show that the Lagrangian dual problem for (9) is given by

<em>n                               n         n</em>

(10)

1            <em>i</em>=1                 <em>i</em>=1 <em>j</em>=1 subject to 0 ≤ <em>α<sub>i </sub></em>≤ <em>C</em>

<em>n</em>

X

<em>α<sub>i</sub>y<sub>i </sub></em>= 0

<em>i</em>=1 Task T6: (5 p) Use complementary slackness (of the KKT conditions) to show that support vectors with <em>y<sub>i</sub></em>(<strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub></em>+ <em>b</em>) <em>&lt; </em>1 have coefficient <em>α<sub>i </sub></em>= <em>C</em>.

<h1>3           Dimensionality reduction on MNIST using PCA</h1>

In the experimental part of this assignment you shall work with the MNIST image data set of handwritten digits, containing data-target pairs with images <strong>X</strong><em><sub>i </sub></em>∈ [0<em>,</em>1]<sup>28×28 </sup>and targets <em>t<sub>i </sub></em>∈ {0<em>,</em>1<em>,…,</em>9}. You will only consider a part of the data set, specifically only images with targets <em>t<sub>i </sub></em>∈ {0<em>,</em>1}. The dataset MNIST_01.mat contains 12665 training datatarget pairs {train_data, train_labels } and 2115 test data-target pairs {test_data, test_labels }, where for both sets the images have been stacked column-wise, i.e., <strong>x</strong><em><sub>i </sub></em>= vec(<strong>X</strong><em><sub>i</sub></em>) ∈ R<sup>28</sup><sup>2</sup>

The MNIST images are of quite low resolution, but the dimensionality is still quite large, i.e., <em>D </em>= 28<sup>2 </sup>= 784. One way of visualizing a collection of high-dimensional data examples is to use a dimensionality reduction technique. Principal component analysis (PCA) is a method for reducing the dimensionality of a dataset, while retaining as much of the data’s variability as possible. PCA utilizes the singular value decomposition (SVD), which for a matrix of <em>N </em>zero-mean data examples of dimension <em>D</em>, i.e., <strong>X </strong>∈ R<em><sup>D</sup></em><sup>×<em>N</em></sup>, is the factorization:

<strong>X </strong>= <strong>USV</strong><sup>&gt;                                                                                                                          </sup>(11)

where for <em>P </em>= min(<em>D,N</em>); <strong>U </strong>∈ R<em><sup>D</sup></em><sup>×<em>P </em></sup>, <strong>S </strong>= diag(<strong>s</strong>)<em>,</em><strong>s </strong>∈ R<em><sup>P </sup></em>, and <strong>V </strong>∈ R<em><sup>N</sup></em><sup>×<em>P </em></sup>are the left singular vectors, the diagonal matrix of <em>P </em>many singular values, and the right singular vectors, respectively. The dimensionality reduction of <strong>X </strong>to dimension <em>d </em>is then the projection of <strong>X </strong>onto the <em>d </em>left singular vectors with the largest absolute singular values.

Task E1: (5 p) Compute a linear PCA and vizualize the whole training data in <em>d </em>= 2 dimensions. Make sure that the necessary condition to apply PCA is met. Display your results in a plot where the data examples are marked with different markers and colors for each of the two classes. Make the plot clearly interpretable by using legends, adjusting font size, etc.

<h1>4           Clustering of unsupervised data using K-means</h1>

In this section, you shall cluster the MNIST images without using their target values. Kmeans clustering is an unsupervised approach for computing K many clusters by iteratively determining the so-called centroids of each cluster and assigning each sample to a cluster. The centroid of a cluster is defined as the mean location of all samples within the cluster. In the assignment step, each sample gets assigned to the closest centroid. Algorithm 1 summarizes the K-means clustering approach.

Algorithm 1 K-means clustering

Select number of clusters <em>K</em>, convergence threshold , and maximum iterations <em>j</em><sub>max</sub>.

Initialize randomly.

for <em>j </em>= 1<em>,…,j</em><sub>max </sub>do

Step 1: Assign examples to clusters:

<em>d<sup>i </sup></em>= <em>f </em>(<strong>x</strong><em><sub>i</sub>,</em><strong>C</strong><sup>(<em>j</em>−1)</sup>)<em>,i </em>= 1<em>,…,N y<sub>i </sub></em>=

Step 2: Assign new cluster centroids:

, k = 1,…,K

<strong>c</strong>, k = 1,…,K

Check convergence:

if <sub>tol </sub>then return <strong>y</strong>(<em>j</em>)<em>,</em><strong>C</strong>(<em>j</em>)

end if

end for

return <strong>y</strong>(<em>j</em>max)<em>,</em><strong>C</strong>(<em>j</em>max)

Task E2: (20 p) Implement K-means clustering for the training data, using the provided function sketch K_means_clustering.m. Do so by completing the following steps:

<ol>

 <li>Define and implement a distance function <em>f</em><sub>xdist</sub>(<strong>x</strong><em>,</em><strong>C</strong>) which computes a distance of your chosing between a single example and the <em>K </em></li>

 <li>Define and implement a pairwise distance function, <em>f</em><sub>cdist</sub>(<strong>C</strong><em>,</em><strong>C</strong><strong><sup>˜</sup></strong>), which computes a distance of your choosing between two cluster centroids, such that the distance is exactly zero when the pairs of centroids agree.</li>

 <li>Construct the first step of the K-means clustering algorithm: Define and implement a function step_assign_cluster which assigns each sample to one of the <em>K </em></li>

 <li>Construct the second step of the K-means clustering algorithm: Define and implement a function step_compute_mean which assigns new centroids by computing the means of the newly assigned clusters, outputting both the new centroids and a measure of the distance which the centroids have moved. The latter will be used as a stopping criterion; the algorithm will stop when the centroids have changed less than a specificed threshold.</li>

 <li>Implement a fully running K_means_clustering using the components you’ve implemented above, and run the algoritm on the original training data, for <em>K </em>= 2 and <em>K </em>= 5 Reuse the pre-trained linear PCA from Task E1 only to visualize your results in two dimensions for each <em>K </em>in a plot where the data examples are marked with a different marker and color for each cluster. Make the plot clearly interpretable by using legends, adjusting font size, etc.</li>

</ol>

Important:

<ul>

 <li>Use the provided function sketch K_means_clustering.m. Do not implement the function from scratch.</li>

 <li>Only make changes to K_means_clustering.m in the lines between % CHANGE and % DO NOT CHANGE. Do not change other parts of the function.</li>

 <li>Explain why the clusters seem to overlap for, e.g., <em>K </em>= 5.</li>

</ul>

We could now check only a few samples from each cluster to assign a suitable label to the centroid. Having assigned a label to a centroid, we assign the same label to all samples in the corresponding cluster. Alternatively, we can plot the centroid image and assign a suitable class manually. These approaches are advantageous if labeling of all samples individually is expensive. Here, we have the privileged situation to know target labels for all samples, so we may also assign the label to a centroid that is the most frequent in the cluster. Assigning the same label to all samples in the cluster will lead to some misclassifications. By summing the number of misclassifications in each cluster, a missclassification rate may be calculated for the dataset.

Task E3: (5 p) Display the <em>K </em>= 2 and <em>k </em>= 5 centroids as images. Make a plot with 1 × <em>K </em>subplots illustrating the centroids from each cluster as an image, using the Matlab function imshow(). Make legends, titles or otherwise mark which cluster is displayed. Note that you have to stack the centroids back into the shape <strong>X</strong><em><sub>i </sub></em>∈ R<sup>28×28 </sup>to display the image properly, using, e.g., reshape().

Task E4: (5 p) Now you shall use K-means clustering for classification, by completing the following steps:

<ol>

 <li>For <em>K </em>= 2, implement a function K_means_classifier which assigns to a given example the label of the closest centroid. You may use the distance function calculated in Task E2. Then assign each cluster centroid the label of which it has the most examples of in the training data.</li>

 <li>Evaluate how many misclassifications occur for the train and test set train_data and test_data by comparing the computed cluster labels to the true labels in train_labels and test_labels. Then produce and fill in the following table (which is provided in the supplementary material):</li>

</ol>

Tabell 1: K-means classification results

<table width="494">

 <tbody>

  <tr>

   <td width="102">Training data</td>

   <td colspan="3" width="282">Cluster           # ’0’        # ’1’         Assigned to class</td>

   <td width="110"># misclassified</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="3" width="282">         1                ?              ?                          ?</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="3" width="282">         …               …            …                         …</td>

   <td width="110">…</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="3" width="282">         K                ?              ?                          ?</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>train = ?</td>

   <td colspan="3" width="282">Sum misclassified:</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="3" width="282">Misclassification rate (%):</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102">Testing data</td>

   <td width="70">Cluster</td>

   <td width="39"># ’0’</td>

   <td width="173"># ’1’             Assigned to class</td>

   <td width="110"># misclassified</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="70">1</td>

   <td width="39">?</td>

   <td width="173">       ?                          ?</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="70">…</td>

   <td width="39">…</td>

   <td width="173">       …                         …</td>

   <td width="110">…</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="70">K</td>

   <td width="39">?</td>

   <td width="173">       ?                          ?</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>test = ?</td>

   <td width="70"> </td>

   <td width="39"> </td>

   <td width="173">Sum misclassified:</td>

   <td width="110">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="70"> </td>

   <td width="39"> </td>

   <td width="173">Misclassification rate (%):</td>

   <td width="110">?</td>

  </tr>

 </tbody>

</table>

Task E5: (5 p) Try out some different <em>K </em>values. Can you lower the misclassification rate further on test data by considering a different number of clusters <em>K</em>?

<h1>5           Classification of MNIST digits using SVM</h1>

In the previous section, you implemented a classifier using unsupervised data, by virtue of the data structure itself. In this section, you shall consider a supervised classifier, namely the support vector machine (SVM). Your task is to use the soft-margin SVM for binary classification, which solves the optimization problem

(12)

<strong>w</strong>

subject to

where <em>y<sub>i </sub></em>= 1 for target <em>t<sub>i </sub></em>= 1, and <em>y<sub>i </sub></em>= −1 for target <em>t<sub>i </sub></em>= 0.

Task E6: (5 p) Use the supervised training data {training_data, training_labels} to train a linear SVM classifier:

<ol>

 <li>One trains the soft-margin SVM by solving the Langrangian dual problem derived in Task T5 using numerical methods. Train a binary soft-margin SVM using Matlab’s built-in function fitcsvm(X,T) on the training data train_data and targets train_labels, where X denotes the <em>N </em>×<em>D </em>matrix of data examples, and T the target vector of length <em>N</em>. Note that the data provided is of size <em>D </em>× <em>N</em>, so you need to transpose it to match the input format. The parameter <em>C </em>in the SVM problem has to be set. The Matlab implementation uses a default value of <em>C </em>= 1<em>.</em>0, which you may leave unchanged.</li>

 <li>Calculate class predictions using the built-in Matlab function predict(model,X), where model is the output from fitcsvm(), and X is the <em>N </em>× <em>D </em>dataset matrix to do predictions on. Then evaluate the misclassification rate on both the training and test data by filling in the following table (which is provided in the supplementary material):</li>

</ol>

Tabell 2: Linear SVM classification results

<table width="392">

 <tbody>

  <tr>

   <td width="102">Training data</td>

   <td width="110">Predicted class</td>

   <td width="139">True class:              # ’0’</td>

   <td width="40"># ’1’</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’0’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’1’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>train = ?</td>

   <td colspan="2" width="250">Sum misclassified:</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="2" width="250">Misclassification rate (%):</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102">Testing data</td>

   <td width="110">Predicted class</td>

   <td width="139">True class:              # ’0’</td>

   <td width="40"># ’1’</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’0’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’1’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>test = ?</td>

   <td colspan="2" width="250">Sum misclassified:</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="2" width="250">Misclassification rate (%):</td>

   <td width="40">?</td>

  </tr>

 </tbody>

</table>

Next, you shall look at SVM with kernels, more specifically a Gaussian kernel. The kernel SVM solves the SVM optimization problem not on the data directly, but in a feature space <strong>z </strong>= <em>φ</em>(<strong>x</strong>), such that the classifier is linear in <strong>z</strong>. In practice, this is done using the so called kernel trick, such that data is never mapped into this feature space explicitly, but by modifying the dual problem appropriately.

Task E7: (10 p) Use the supervised data again to train a non-linear kernel SVM classifier, using a Gaussian kernel:

<ol>

 <li>Train an SVM with Gaussian kernel using Matlab’s built-in function, by specifying fitcsvm(X,T,’KernelFunction’,’gaussian’).</li>

 <li>The Gaussian <u>ker</u>nel has a scaling parameter <em>σ</em><sup>2</sup>, which in Matlab’s SVM estimator is set to <em>β </em>= <sup>p</sup>1<em>/σ</em><sup>2 </sup>= 1 by default, but may be modified by specifying fitcsvm(X,T, ’KernelFunction’,’gaussian’, ’KernelScale’,beta) for some value beta. Can you lower the misclassification rate on the test data by tuning beta?</li>

 <li>Present your best results, similar to Task E5, by filling in the following table (which is provided in the supplementary material):</li>

</ol>

Tabell 3: Gaussian kernel SVM classification results

<table width="392">

 <tbody>

  <tr>

   <td width="102">Training data</td>

   <td width="110">Predicted class</td>

   <td colspan="2" width="179">True class:           # ’0’        # ’1’</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’0’</td>

   <td colspan="2" width="179">                                   ?                  ?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’1’</td>

   <td colspan="2" width="179">                                   ?                  ?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>train = ?</td>

   <td colspan="3" width="290">                                         Sum misclassified:               ?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="3" width="290">                         Misclassification rate (%):                ?</td>

  </tr>

  <tr>

   <td width="102">Testing data</td>

   <td width="110">Predicted class</td>

   <td width="139">True class:              # ’0’</td>

   <td width="40"># ’1’</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’0’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td width="110">’1’</td>

   <td width="139">?</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"><em>N</em>test = ?</td>

   <td colspan="2" width="250">Sum misclassified:</td>

   <td width="40">?</td>

  </tr>

  <tr>

   <td width="102"> </td>

   <td colspan="2" width="250">Misclassification rate (%):</td>

   <td width="40">?</td>

  </tr>

 </tbody>

</table>

Task E8: (5 p) We can achieve very low misclassification rate on both train and test data with a good choice of parameters for the Gaussian kernel SVM. Can we therefore expect the same error on new images? Explain why such a conclusion would be false.

<h1>A           Supplementary material</h1>

Contents of data archive A2_data.mat used in sections 3-5:

train_data_01 MNIST images of ’0’ and ’1’ for training, <strong>X </strong>∈ R<sup>784×12665 </sup>train_labels_01 MNIST labels ’0’ and ’1’ for training, <strong>t </strong>∈ R<sup>12665 </sup>test_data_01       MNIST images of ’0’ and ’1’ for testing, <strong>X </strong>∈ R<sup>784×2115 </sup>test_labels_01             MNIST labels ’0’ and ’1’ for testing, <strong>t </strong>∈ R<sup>2115</sup>

Provided files:

<ul>

 <li>m</li>

</ul>

Matlab function sketch [y,C] = K_means_clustering(X,K), returns cluster assignments y and cluster centroids C given an input of data X the number of clusters K.

<ul>

 <li>tex:</li>

</ul>

L<sup>A</sup>TEX-formatted tables from Section 3 and 4, to be filled in and used in the report.