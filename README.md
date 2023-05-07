Download Link: https://assignmentchef.com/product/solved-pml-homework-06
<br>
Let’s consider a binary classification problem on Half Moons dataset, which consists of two interleaving half circles. The input is two-dimensional and the response is binary (0,1).

We observe 100 points <em>x </em>from this dataset and their labels <em>y</em>:

<strong>1.0.1        scikit-learn GaussianProcessClassifier</strong>

<ol>

 <li>GaussianProcessClassifier from scikit-learn library [1] approximates the non-Gaussian posterior by a Gaussian using Laplace approximation. Define an RBF kernel kernels.RBF with lenghtscale parameter = 1 and fit a Gaussian Process classifier to the observed data (x,y).</li>

</ol>

[ ]:

<ol start="2">

 <li>Use plot_sklearn_predictions function defined below to plot the posterior predictive mean function over a finite grid of points. You should pass as inputs the learned GP classifier sklearn_gp_classifier, the observed points x and their labels y.</li>

</ol>

[25]: <strong>def </strong>meshgrid(x, n, eps=0.1): x0, x1 = np.meshgrid(np.linspace(x[:, 0].min()-eps, x[:, 0].max()+eps, n),

np.linspace(x[:, 1].min()-eps, x[:, 1].max()+eps, n))

x_grid = np.stack([x0.ravel(), x1.ravel()], axis=-1) <strong>return </strong>x0, x1, x_grid

<strong>def </strong>plot_sklearn_predictions(sklearn_gp_classifier, x, y): x0, x1, x_grid = meshgrid(x, 30)

preds = sklearn_gp_classifier.predict_proba(x_grid) preds_0 = preds[:,0].reshape(x0.shape)

<strong>1.0.2 Pyro classification with HMC inference </strong>Consider the following generative model

<em>y<sub>n</sub></em>|<em>p<sub>n </sub></em>∼ Bernoulli(<em>p<sub>n</sub></em>)                <em>n </em>= 1<em>,…,N</em>

logit

LogNormal(0<em>,</em>1)

LogNormal(0<em>,</em>1)

We model the binary response variable with a Bernoulli likelihood. The logit of the probability is a Gaussian Process with predictors <em>x<sub>n </sub></em>and kernel matrix <em>K<sub>σ,l</sub></em>, parametrized by variance <em>ρ </em>and lengthscale <em>l</em>.

We want to solve this binary classification problem by means of HMC inference, so we need to reparametrize the multivariate Gaussian GP(<em>µ,K<sub>σ,l</sub></em>(<em>x<sub>n</sub></em>)) in order to ensure computational efficiency. Specifically, we model the logit probability as

logit(p) = <em>µ </em>· 1<em><sub>N </sub></em>+ <em>η </em>· <em>L,</em>

where <em>L </em>is the Cholesky factor of <em>K<sub>σ,l </sub></em>and <em>η<sub>n </sub></em>∼ N(0<em>,</em>1). This relationship is implemented by the get_logits function below.

[6]: <strong>def </strong>get_logits(x, mu, sigma, l, eta): kernel = gp.kernels.RBF(input_dim=2, variance=torch.tensor(sigma),␣

<em>,</em><sub>→</sub>lengthscale=torch.tensor(l))

<ul>

 <li>= kernel.forward(x, x) + torch.eye(x.shape[0]) * 1e-6</li>

 <li>= K.cholesky() <strong>return </strong>mu+torch.mv(L,eta)</li>

</ul>

<ol start="3">

 <li>Write a pyro model gp_classifier(x,y) that implements the reparametrized generative model, using get_logits function and plate on independent observations.</li>

</ol>

[ ]:

<ol start="4">

 <li>Use pyro NUTS on the gp_classifier model to infer the posterior distribution of its parameters. Set num_samples=10 and warmup_steps=50. Then extract the posterior samples using pyro .get_samples() and print the keys of this dictionary using .keys()</li>

</ol>

[ ]:

The posterior_predictive function below outputs the prediction corresponding to the <em>i</em>-th sample from the posterior distribution. plot_pyro_predictions calls this method to compute the average prediction on each input point and plots the posterior predictive mean function over a finite grid of points.

<ol start="5">

 <li>Pass the learned posterior samples obtained from NUTS inference to plot_pyro_predictions and plot the posterior predictive mean.</li>

</ol>

[ ]:

<h2>1.1          References</h2>

<ul>

 <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html">sklearn GP classifier</a></li>

 <li><a href="https://pyro.ai/examples/gp.html">pyro GPs</a></li>

</ul>