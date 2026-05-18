# Speaker Notes: Neural Differential Cryptanalysis

**Presentation Time Limit:** ~15 minutes
**Target Audience:** Academics, Course Instructors, Peers
**Tone:** Intuitive, data-driven, grounded in first principles. Focus heavily on *why* instead of just *what*.

---

## Slide 1: Title Slide
**Speaker Notes:**
- "Good morning/afternoon everyone. Today, I'll be presenting my course project on Neural Differential Cryptanalysis, focusing specifically on a systematic study of the SPECK32/64 cipher."
- "The goal of this research is not just to build another AI that breaks a cipher, but to aggressively unpack *how* it's doing it—breaking down the black box to understand the actual cryptographic signals it relies on."

**Potential Q&A:**
- *Q: Why choose SPECK32/64?*  
  **A:** SPECK is an ARX cipher (Addition, Rotation, XOR) that doesn't use S-boxes. This makes it structurally minimal and perfect for a first-principles study on how neural networks learn algebraic cryptographic properties, rather than just memorizing lookup tables.

---

## Slide 2: Outline
**Speaker Notes:**
- "We'll start by establishing the motivation—what AI brings to classical cryptanalysis. Then we'll build a quick intuitive background on differentials and SPECK."
- "From there, I’ll lay out our 3 core research questions, walk through the 15 controlled experiments we ran, and conclude with the key insights we extracted about representations, Markov properties, and cipher structures."

---

## Slide 3: Can Neural Networks Break Ciphers?
**Speaker Notes:**
- "Let's build from first principles. Classical differential cryptanalysis relies on human intuition. Analysts manually build propagation tables and carefully piece together bit-trails to find statistical biases. Fast forward to higher rounds, and the search space explodes."
- "In 2019, Gohr introduced a paradigm shift: use a neural network as a 'distinguisher'. Rather than feeding it manually crafted rules, he simply gave it raw ciphertext pairs, and it reached 92% accuracy on 5 rounds."
- "But this left a massive open question: *What exactly is it learning?* Is it memorizing data, or is it learning genuine cryptographic structure? That is what this project answers."

**Potential Q&A:**
- *Q: What do you mean by a "distinguisher"?*  
  **A:** A distinguisher is simply an algorithm—or in this case, a neural net—that can reliably tell the difference between raw random noise and actual encrypted structure. If it guesses correctly more than 50% of the time, the cipher is effectively broken.

---

## Slide 4: Differential Cryptanalysis & SPECK32/64
**Speaker Notes:**
- "To understand what the neural network is looking for, we have to understand the fundamental mechanism: the differential."
- "If we take two plaintexts, $P$ and $P'$, that differ by exactly $\Delta P$, the cipher encrypts them into $C$ and $C'$. The probability that they output a specific difference $\Delta C$ is the differential probability."
- "Classically, we assume the Markov property: that the probability of a multi-round trail is just the product of the single-round probabilities—meaning no 'memory' is passed beyond the current state."
- "We apply this to SPECK32/64. It’s an incredibly simple ARX structure—just XOR, bit-shifts, and modular addition."

**Potential Q&A:**
- *Q: Why evaluate starting at $\Delta P =$ 0x0040_0000?*  
  **A:** This is a mathematically proven, optimal input difference for SPECK32. Starting with a known good difference ensures we are actually measuring the network's ability to trace real signals rather than failing due to a bad initialization.

---

## Slide 5: Neural Distinguishers
**Speaker Notes:**
- "So, how do we formulate this for a neural network? We treat it as binary classification."
- "Class 1: Here are two ciphertexts generated from the same carefully crafted input difference. Class 0: Here are two completely random uniformly generated numbers. The network's job is to tell them apart using cross-entropy loss."
- "If it scores 50\%, it's guessing. Any accuracy meaningfully above 50\% means it's finding a non-linear decision boundary that exposes a vulnerability in the cipher."

---

## Slide 6: Research Questions & Experimental Setup
**Speaker Notes:**
- "We designed an extensive pipeline of 15 experiments to answer three questions:"
- "1. Representation: Does how we format the data matter?"
- "2. Markov: Does the classical memoryless assumption hold true under the scrutiny of a neural network?"
- "3. Architecture: What is the most efficient way to extract this signal?"
- "We didn't just train one model. We tested 7 data formats, 6 architectures, balanced 10 million pairs per run, and wrapped everything in 5-seed statistical bootstrapping to ensure rigor."

---

## Slide 7: Baseline: The Distinguishing Cliff
**Speaker Notes:**
- "Here is our baseline reality. As you can see, rounds 3 and 4 are trivial for the network—close to 100\%."
- "Round 5 sits safely at 87.4\%. But then we hit what I call the 'distinguishing cliff'. By round 7, the network completely collapses to 50\%, exactly equal to random guessing."
- "The physical diffusion of the cipher outpaces the network's capacity to trace the signal. Our immediate follow-up was: what drives this limit?"

---

## Slide 8: Representation Dominates Architecture
**Speaker Notes:**
- "The single most surprising finding of our research is this: *How* you show the data to the network matters exponentially more than *how big* the network is."
- "Look at 5 rounds: Simply XORing the ciphertexts before passing them to the network (R2) yields an 88.6\% accuracy. But feeding raw word-level blocks (R5) collapses the accuracy to 59.8\%. That's a 28-point gap!"
- "Conversely, swapping from a basic MLP to an advanced LSTM only yields a 3.8 point gain. Precomputing the XOR difference halves the dimensionality of the search space, effectively giving the network a 'free lunch'."

**Potential Q&A:**
- *Q: Why does R2 (XOR) work so well if the cipher uses modular addition (which is non-linear w.r.t XOR)?*  
  **A:** Even though modular addition creates complex carry chains, the XOR difference still captures the linear approximations of those operations incredibly well. The neural net basically uses the XOR difference as a highly correlated foundation and learns to correct the non-linear noise itself.

---

## Slide 9: Signal Decay Heatmap
**Speaker Notes:**
- "To prove that the network follows the differential, we plotted a heatmap of accuracy across different starting differences."
- "Notice the decay: If we start with the optimal difference (0x0040_0000), we hold on to 86\% accuracy at round 5. If we start with a poor difference, like 0x01, the signal dies instantly."
- "This proves the AI isn't finding 'magic' backdoors; it is strictly constrained by the mathematical diffusion properties of the ARX operations."

---

## Slide 10: The Network Discovers Cipher Structure
**Speaker Notes:**
- "Now, we open the black box. We computed the gradient saliency—basically asking the network, 'which bits are you looking at to make your decision?'"
- "The network isn't looking everywhere. It focuses intensely on bits 14, 28, 12, 13, and 29."
- "Intuitively, this maps perfectly to SPECK's DNA! Bit 14 shifted by 7 positions becomes bit 7—the exact boundary of the modular addition carry chain. To prove this wasn't a fluke, we randomly permuted the bits. The accuracy instantly dropped 37 points. It truly learned position-specific structure."

**Potential Q&A:**
- *Q: Could the model just be looking at the Hamming weight?*  
  **A:** No, that's exactly what the invariance test disproves. If it were just counting 1s and 0s (Hamming weight), scrambling the order wouldn't change the count, and accuracy would remain high. The 37-point drop proves it relies on the *physical location* of the bits.

---

## Slide 11: Attention Interpretation
**Speaker Notes:**
- "We verified this using a Transformer architecture to visualize attention entropy."
- "As we increase the rounds and the cipher gets harder, the attention entropy strictly decreases. Meaning, instead of panicking and looking everywhere, the model *hyper-focuses* on the few surviving bit positions that still carry signal. It perfectly corroborates the saliency map."

---

## Slide 12: Robustness Under Noise
**Speaker Notes:**
- "In practical side-channel attacks, data is never clean. We subjected our distinguisher to noise."
- "It handles Gaussian noise fairly well, but bit-flips are completely destructive. Even flipping a single bit (1\%) drops accuracy by over 3 points. At a 5\% Bit Error Rate, the distinguisher goes deaf."
- "The intuition here is simple: cryptographic diffusion means flipping one bit creates a cascade effect that destroys the tightly coupled relationships the network is tracking."

---

## Slide 13: Markov Assumption: Validated
**Speaker Notes:**
- "Our next major question: Classically, we assume transitions are Markovian—memoryless. Does a neural network exploit hidden multi-round memory?"
- "We checked this two ways. First, depth analysis showed that each round scales the bias multiplicatively, exactly as Markov expects. Second, we measured conditional mutual information between rounds, and found literally zero residual dependency."
- "We can confidently state: For SPECK32, the classical Markov assumption strictly holds."

**Potential Q&A:**
- *Q: How exactly did you measure Mutual Information?*  
  **A:** We trained a MINE (Mutual Information Neural Estimator) which provides a differentiable, scalable way to approximate the KL-divergence between the joint and marginal distributions of the round states.

---

## Slide 14: Latent Space & Generative Markov Validation
**Speaker Notes:**
- "As a final nail in the Markov coffin, we built a Variational Autoencoder to generate the distributions."
- "If you look at the latent space, the VAE cleanly separates the real cipher text vectors from random ones in 2D space. More importantly, when we tried to explicitly force the model to 'remember' past rounds using memory vectors, the loss ratio shot above 1.0."
- "Adding memory literally made the model perform worse, confirming there is no hidden long-term dependency to exploit."

---

## Slide 15: Neural vs Classical & Automated Search
**Speaker Notes:**
- "So, why use neural nets at all if they follow classical rules? Efficiency."
- "Using classical statistics, detecting a 5-round signal requires analyzing roughly 2-to-the-16 power pairs. At our sample size, classical methods report 0.00\% confidence."
- "Meanwhile, the neural network hits 87.8\%. It is phenomenally more data-efficient at extracting marginal biases because it learns non-linear relationships that human-crafted tables miss."

---

## Slide 16: RL-Based Differential Search
**Speaker Notes:**
- "If neural networks can detect signals, can they *find* them autonomously?"
- "We flipped the problem and used Reinforcement Learning to search the input difference space. At 3 rounds, the RL agent autonomously discovered an optimal differential (84.9\%) starting from scratch."
- "However, by 5 rounds, the gradient signal is too sparse, and the RL fails to explore. It shows both the incredible potential for automated cryptanalysis, and the current boundaries of gradient-based search."

---

## Slide 17: Architecture & Data Efficiency
**Speaker Notes:**
- "As for architectures, simple is better. An LSTM squeezed out 90\% accuracy using 600K parameters, but a basic 1D Convolutional Network achieved 88.4\% with almost 8-times fewer parameters."
- "Data-wise, the model crosses the 75\% threshold with less than a thousand examples. Supplying 1 million pairs gives diminishing returns. The network rapidly learns the linear approximations, but struggles to learn the deeper non-linear noise."

---

## Slide 18: No Cross-Round Transfer
**Speaker Notes:**
- "Our final experiment was zero-shot transfer learning. We took a master model trained on 5 rounds and tested it on 3 and 4 rounds. It scored 45\%."
- "Let me repeat that: it scored *worse than random guessing*."
- "Why? Because features at 5-rounds are literally anti-correlated with the features at 3-rounds. The network doesn't learn a generic 'cipher detector'; it meticulously overfits to the exact algebraic permutations of a specific round depth."

**Potential Q&A:**
- *Q: Why is it anti-correlated rather than just 50% (random)?*  
  **A:** Because of the rotation and XOR operations. A bit position that is highly indicative of a real cipher-pair at round 3 gets shifted and XORed such that by round 5, its value might be exactly inverted or phase-shifted. The network expects the round-5 configuration, gets the round-3 configuration, and confidently guesses 'random' (class 0) for what is actually a real cipher pair!

---

## Slide 19: Key Findings
**Speaker Notes:**
- "To summarize everything from first principles:"
- "One: Representation reigns supreme. Do the easy math (XOR) for the network so it can focus on the hard math."
- "Two: The Markov assumption survives the neural era. The limits of deep learning here are bounded by traditional probability."
- "Three: The network isn't black magic. It learns hyper-specific, position-dependent relationships tied directly to the cipher's rotation constants."

---

## Slide 20: Contributions & Future Directions
**Speaker Notes:**
- "Our pipeline gives a fully open-source, unit-tested framework to rigorously break down neural distinguishers."
- "Moving forward, the focus must shift strictly to extending this 7-round barrier using customized deep architectures, and transitioning from finding 'distinguishers' to full key-recovery attacks."

---

## Slide 21: Thank You
**Speaker Notes:**
- "Thank you for your time and attention. Our codebase and full reproducibility suite is available on GitHub. I'd be happy to take any questions."
