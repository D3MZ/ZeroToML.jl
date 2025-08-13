# Character-Level Tokenizer

This document outlines the process of converting text into a numerical format suitable for machine learning models and back. The process consists of three main steps: building a vocabulary, encoding text into integers, and decoding integers back into text.

## 1. Vocabulary Building

The first step is to create a vocabulary from a given corpus of text. The vocabulary is the set of all unique characters present in the text.

Given an input text $T$, the vocabulary $V$ is constructed as follows:
1.  Collect all characters from the text $T$.
2.  Find the set of unique characters.
3.  Sort these unique characters to ensure a consistent mapping.

$$
V = \mathrm{sort}(\mathrm{unique}(\mathrm{characters}(T)))
$$

For example, given the text $T = \text{"hello world"}$:
- Unique characters are: 'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'.
- Sorted vocabulary $V$: `[' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']`.

## 2. Encoding

Encoding is the process of converting a string of text into a sequence of integers using the vocabulary.

First, a mapping from each character in the vocabulary to its integer index is created. Let's call this map `char_to_int`.

$$
\mathrm{char\_to\_int}: V \to \{1, 2, \dots, |V|\}
$$
where `char_to_int`$(v_i) = i$ for the $i$-th character $v_i$ in $V$.

The input text $T$ is then transformed into a sequence of integers $E$ by looking up each character of $T$ in the `char_to_int` map.

$$
E = [\mathrm{char\_to\_int}(c) \text{ for } c \in T]
$$

Using the example text $T = \text{"hello world"}$ and its vocabulary $V$:
- The encoded sequence is $E = [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]$.

## 3. Decoding

Decoding reverses the encoding process, converting a sequence of integers back into a string.

Given an encoded sequence $E$ and the vocabulary $V$, the original text $T$ is reconstructed by mapping each integer in $E$ back to its corresponding character in $V$.

$$
T = \mathrm{join}([V[i] \text{ for } i \in E])
$$

For the encoded sequence $E = [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]$ and the vocabulary $V$:
- Joining the characters corresponding to each index results in the original text: "hello world".
