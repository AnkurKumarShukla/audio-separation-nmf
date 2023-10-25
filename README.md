# audio-separation-nmf
logic code 
https://www.kaggle.com/code/ankurkumarshukla/202051029-nmf-ankur

This project aims to seperate audio file with multiple sources. It take an audio file as input and return S audio file (where S is number of sources want to separate) .

## Description

# Non-Negative Matrix Factorization (NMF) for Audio Separation

Non-Negative Matrix Factorization (NMF) is a powerful technique used in audio processing for separating mixed audio sources into their constituent parts. Here's a step-by-step explanation of how it works:

1. **Audio Signal Representation**: The input is a mixed audio signal that combines multiple sources, like vocals, music, and noise. This signal can be represented as a matrix with time frames as rows and frequency components as columns.

2. **NMF Model**: NMF aims to decompose the mixed audio matrix into two non-negative matrices, 'W' and 'H'. 
   - 'W' represents the basis vectors or templates that are non-negative and capture spectral characteristics of the sources.
   - 'H' represents the activation matrix, showing how much of each source is present in the mixture at each time frame.

3. **Matrix Factorization with Multiplicative Update Rule**: NMF is typically performed using a multiplicative update rule to minimize a cost function, often the Frobenius norm or Kullback-Leibler divergence. The multiplicative update rule iteratively updates 'W' and 'H' until convergence:
   - Initialize 'W' and 'H' with non-negative values.
   - Iteratively update 'W' and 'H' as follows:
     - Update 'W': `W_new = W * ((X * H^T) / (W * H * H^T))`, where 'X' is the mixed audio matrix.
     - Update 'H': `H_new = H * ((W^T * X) / (W^T * W * H))`.

4. **Convergence**: The algorithm repeats the update steps until 'W' and 'H' converge to approximate the original mixed signal matrix as closely as possible.

5. **Source Separation**: Once 'W' and 'H' have converged, you can use them to reconstruct the separated audio sources. The separation process typically involves multiplying 'W' and 'H' to obtain an estimate of the mixed signal, which can then be used to isolate specific sources.

NMF for audio separation is a fundamental technique in tasks like extracting vocals from music tracks or isolating specific sounds in audio recordings. The multiplicative update rule ensures that the factorization is performed with non-negative values, which aligns with the assumption that audio sources are non-negative.

You can apply this technique to various audio separation tasks, and it's widely used in music production, speech processing, and more.


## Getting Started

### Dependencies

* Refer requirement.txt
* ex. Windows 10

### Installing


* Python 3.x
* Can  directly go to deployed link to have demo
### Executing program

* Fork , clone then run `python model.py`
* API will be activate
```
Happy coding
```

## Help


```
contact me via linkdin or mail
```

## Authors

Ankur Shukla
[@linkedin](https://www.linkedin.com/in/ankur-shukla-iiitv/)



## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## Acknowledgments

Inspiration
* [PAPER NIPS](https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
