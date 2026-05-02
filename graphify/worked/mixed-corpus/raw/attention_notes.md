# Attention Mechanism Notes

Notes on the Transformer architecture from Vaswani et al., 2017.
arXiv: 1706.03762

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. The Transformer is a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

## Multi-Head Attention

The model uses h=8 parallel attention heads. For each head, d_k = d_v = d_model/h = 64.

Scaled dot-product attention:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-head attention runs h attention functions in parallel, then concatenates and projects:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

The scaling by sqrt(d_k) prevents the dot products from growing large in magnitude, which would push the softmax into regions with very small gradients.

## Architecture

The Transformer uses a stacked encoder-decoder structure.

Encoder: 6 identical layers, each with two sublayers:
1. Multi-head self-attention
2. Position-wise fully connected feed-forward network

Each sublayer uses a residual connection followed by layer normalization:
    output = LayerNorm(x + Sublayer(x))

Decoder: 6 identical layers, each with three sublayers:
1. Masked multi-head self-attention (prevents positions from attending to subsequent positions)
2. Multi-head attention over encoder output
3. Position-wise feed-forward network

d_model = 512 for all sublayers and embedding layers.
Feed-forward inner dimension = 2048.

## Positional Encoding

Since the model contains no recurrence and no convolution, positional encodings are added to the input embeddings to give the model information about the relative position of tokens:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This allows the model to easily learn to attend by relative positions.

## Why attention over recurrence

Three main advantages:
1. Total computational complexity per layer is lower for self-attention when sequence length is smaller than representation dimensionality
2. Computations that can be parallelized — recurrent layers require O(n) sequential operations
3. Path length between long-range dependencies is O(1) for self-attention vs O(n) for recurrence

## Results

WMT 2014 English-to-German: 28.4 BLEU, outperforming all previously published results by over 2 BLEU.
WMT 2014 English-to-French: 41.0 BLEU, new state of the art.
Training cost: 3.5 days on 8 P100 GPUs.

## Open questions

[1] Does the choice of h=8 heads generalize, or is it architecture-specific?
[2] The scaling factor sqrt(d_k) is justified empirically — is there a theoretical justification?
[3] How does learned positional encoding compare to sinusoidal at longer sequence lengths?

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. arXiv:1706.03762
[2] Ba, J., Kiros, J., Hinton, G. (2016). Layer Normalization. arXiv:1607.06450
[3] He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
