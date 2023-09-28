---
layout: layout
title: Transformers
---
# Introduction to Attention

Consider the following two sentences:

<center>1. <i>The dog did not cross the road as <u>it</u> was too tired.</i></center>
<center>2. <i>The dog did not cross the road as <u>it</u> was too narrow.</i></center>

In the first sentence, the pronoun _it_ refers to dog whereas in the second sentence _it_ refers to the road. We need our model to understand this relationship. To achieve this, we can use the whole sequence to compute a _weighted average_ of each embedding instead of using a fixed embedding for each token. Another way to formulate this is to say that given a sequence of token embeddings $$x_1, \ldots, x_n$$,  we produce a sequence of new embeddings $$x_1', \ldots, x_n'$$ where each $$x_i'$$ is a linear combination of all the $$x_j$$:

$$
x_i' = \sum_{j = 1}^{n} w_{ji}x_j
$$

The coefficients $$w_{ji}$$ are called _attention weights_ and are normalized so that $$\sum_j w_{ji} = 1$$. The weighted averaging scheme would probably assign a higher weight to the word _dog_ when creating the embedding for the word _it_ for our first sentence example and would assign a higher weight on _road_ for the second one. Embeddings that are generated in this way are called _contextualized embeddings_ and predate the invention of transformers in language models like ELMo.

<blockquote style="background-color: #FFFFE0; padding: 10px;">
<b>
Note: We use the terms <i>token</i> and <i>word</i> interchangeably here, even though they could be different. More on this later.
</b>
</blockquote>

# Computing the Attention Weights

## Dot-Product (Multiplicative)

The dot product-based scoring function is the simplest one and has no parameters to tune. We can compute the dot product between $$x_i$$ and $$x_j$$ to compute $$w_{ij}$$. Dot product indicates how similar $$x_i$$ is to $$x_j$$ and can hence be used as a weight.

$$
w_{ji} = x_i \cdot x_j
$$

## Scaled Dot-Product Attention

The scaled dot product-based scoring function divides the dot product by the square root of the dimension. According to Vaswani et al., as the dimension increases, the dot products grow larger, which pushes the softmax function into regions with extreme gradients.

$$
w_{ji} = \frac{x_i \cdot x_j}{\sqrt{d}}
$$

To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $$x_i \cdot x_j = \sum_{r=1}^{d} x_{ir} x_{jr}$$, has mean $0$ and variance $$d$$.


In our simple example, we only used the embeddings “as is” to compute the attention scores and weights. Though this mechanism is simple, there is no learning of weights happening in this. And this is where the mechanism of query and key matrices comes into the picture.  In practice, the self-attention layer applies three independent linear transformations to each embedding to generate the query, key, and value vectors. These transformations project the embeddings and each projection carries its own set of learnable parameters, which allows the self-attention layer to focus on different semantic aspects of the sequence.

# Self-Attention: Queries, Keys and Values

The concept of self-attention is based on three vector representations:

1. Query
2. Key
3. Value

Instead of directly finding the dot product between $$x_i$$ and $$x_j$$, we project each token embedding into three vectors called _query_, _key_, and _value_.

Consider sentence 1 again for example. Let's focus on finding the embedding for the token _it_. Assume that the dimension of all the tokens is 512 i.e. we represent each token through a dense vector of length 512. All these embeddings are randomly initialised before training and are learned during training the model.

Instead of finding the dot product between _it_ and all other tokens in the sentence and using them as the weights, we do the following:

- Step 1: Define a query matrix $$Q$$ of dimension $$512 \times 64$$ and multiply the embedding vector of _it_ with this matrix to project the embedding of _it_ into a $$64$$-dimensional space. Let us call this new vector as _query_.

- Step 2: Define a key matrix $$K$$ of dimension $$512 \times 64$$ and multiply all the token embeddings (including the token _it_) with this matrix. All the tokens are now projected into another $$64$$-dimensional space. Let us call these vectors as _keys_.

- Step 3: Define a value matrix $$V$$ of dimension $$512 \times 64$$ and multiply all the token embeddings (including the token _it_) with this matrix. All the tokens are now projected into another $$64$$-dimensional space. Let us call these vectors $v_1, \ldots, v_n$ as _values_.

- Step 4: Determine how much the query and key vectors relate to each other using a similarity function (scaled dot-product attention). Our attention mechanism with equal query and key vectors will assign a very large score to identical words in the context, and in particular to the current word itself (and hence the name _self-attention_). The outputs from this step are called the attention scores, and for a sequence with $$n$$ input tokens there is a corresponding $$n \times n$$ matrix of attention scores.

- Step 5: Multiply the attention scores by a scaling factor to normalize their variance and then normalized with a softmax to ensure all the column values sum to 1. This is done in order to make it possible to build gradients that are more stable. The resulting $$n \times n$$ matrix now contains all the attention weights, $$w_{ji}$$.

- Step 6: Update the token embeddings. Once the attention weights are computed, we multiply them by the value vector $v_1, \ldots, v_n$ to obtain an updated representation for embedding of the token _it_ $x_i'$:

$$
x_i' = \sum_j w_{ji} v_j
$$

Similarly, we can find the embeddings of all other tokens by casting them into their query vectors using the same query matrix $$Q$$, finding their attention weights using the key vectors and using these weights to compute the weighted average of the value vectors.

Instead of a vector computation for each token $i$, the input matrix $X \in \mathbb{R}^{l \times d}$, where $l$ is the maximum length of the sentence and $d$ is the dimension of the inputs, combines with each of the query, key, and value matrices as a single computation given by:

$$
\text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where $$d_k=64$$ in our case. $$Q, K$$ and $$V$$ weight matrices are randomly initialized and the weights are jointly learned from the training process.

This self-attention layer helps the model capture the context of the word in its representation for example, it may be about learning grammar, tense, conjugation, etc..

# Multi-Headed Attention

Instead of a single self-attention head, there can be $$h$$ parallel self-attention heads; this is known as multi-head attention. Multi-head attention involves multiple sets of query/key/value weight matrices, each resulting in different query/key/value matrices for the inputs. These output matrices from each head are concatenated and multiplied with an additional weight matrix, $W_O$. 

In the original transformer paper, the authors used $$h = 8$$ heads. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

where

$$
\text{head}_i = \text{Attention}(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})
$$

The projections are parameter matrices:


$$
W_{i}^{Q} \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_{i}^{K} \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_{i}^{V} \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$ and $$W_O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$$

For each self-attention head, the authors used $$d_k = d_v = \frac{d_{\text{model}}}{h} = 64$$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

Each head can learn something different, for example, it may be about learning grammar, tense, conjugation, etc. For instance, one head can focus on subject-verb interaction, whereas another finds nearby adjectives. It also helps the model expand the focus to different positions. Obviously we don’t handcraft these relations into the model, and they are fully learned from the data. If you are familiar with computer vision models you might see the resemblance to filters in convolutional neural networks, where one filter can be responsible for detecting faces and another one finds wheels of cars in images.

