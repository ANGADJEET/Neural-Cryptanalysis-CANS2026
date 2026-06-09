# Review of “The Anti-Transfer Paradox” Draft

## Part 1: First Impression

- **Overall Tone:** The paper reads more like a detailed technical report or an extended analysis than a streamlined conference submission. The writing is dense and thorough, but often overly long and elaborative. Key results (the “anti-transfer paradox,” architecture-independence, etc.) are buried in long paragraphs and bullet lists, rather than introduced succinctly. 

- **Conference vs. Report:** It feels closer to **(B)** – a technical report or preprint – rather than a polished conference paper. Indicators include extensive experimental details (many subtables, algorithm pseudo-code, elaborate controls, etc.), several in-depth proofs alluded to (pushed to appendix), and a very long related-works section that wanders into tangential improvements. A conference paper would typically be more concise in background and focus on the core narrative. Here, the narrative is somewhat diffused by minutiae.

- **Polish and Maturity:** The writing is competent but not polished. There are multiple places where phrasing is awkward, paragraphs are overlong, and transitions are weak (discussed further in Part 4). For example, the introduction immediately jumps into “two foundational questions” and detailed contributions without a smooth motivating build-up or an outline of the paper’s structure. Some terminology (like “DDT”) is used without definition. The mix of formal theorem statements and very informal footnoted commentary (“Throughout, ‘pp’ denotes…”) gives a slightly unsteady tone. Overall, it feels like work by knowledgeable researchers but still draft-quality in exposition.

- **Claims vs. Evidence:** There are moments of overclaiming or unnecessary hype. Phrases like “two foundational questions,” “we rigorously prove,” and “the first empirical test” stand out as strong claims. While the claims may be legitimate, they should be toned down or supported by citations. Conversely, there is a fair amount of humility too (e.g. acknowledging that their MLP is a “deliberately weak baseline”). The paper mostly avoids obvious hype like “groundbreaking” or “novel,” but calling questions “foundational” or “paradox” without clear buildup can feel strong. 

- **Rushed?** It does *not* feel rushed in terms of experiments — the authors have clearly done a lot of work. But it does feel like the writing was pieced together around the results. The exposition leaps into results-heavy content without first setting a high-level roadmap. Some sections (like the related work and methodology) feel bitty and assembled, suggesting the authors wrote them to record details rather than craft a narrative flow.

- **Experienced vs. Students:** The depth of experiments, the formal theorem, and the use of advanced tools (MINE, VAE, genetic search) indicate experienced researchers. However, the writing occasionally has the hallmarks of a less-seasoned writer: e.g., overlong sentences, slightly informal asides, and a lack of concise storytelling. If this is students writing under supervision, that would fit; an experienced author would usually trim more aggressively and ensure smoother flow. 

**Exact reasons for these impressions:** The mixture of formal and informal tone, missing elements like an explicit “paper organization,” and very detailed subsections (e.g. **Algorithm 1**, extended experimental controls) gave the impression of a report. The narrative jumps (e.g., contributions list jumping to a formal “Markov–Transfer conjecture” point 4 without finishing on contributions) and occasional awkward phrasing (“≤2.6 pp1” footnote glitch) erode polish. In short, the content quality is high, but the presentation (structure and phrasing) is not yet at top-conference level.

## Part 2: Structural Analysis (Section-by-Section)

### Abstract

- **Purpose:** Summarize the problem, approach, and main results in a compact way. 

- **Assessment:** The abstract is factually informative but overloaded with specifics. It identifies the two questions (signal location and compositionality) and states key findings. However, it immediately dives into detailed quantitative results (exact accuracy percentages, p-values, mutual information figures). This is unusual for an abstract, which should ideally emphasize the *high-level* takeaway rather than every numeric detail.

- **Strengths:** It clearly states the context (“neural differential distinguishers, round-reduced ciphers, three families”) and enumerates the main conclusions (signal in XOR-differences, positive vs. anti-transfer, “anti-transfer paradox”). It does mention the methodology (“cross-round transfer experiments, controlled, ... VAE-based tests”), covering how the questions are answered.

- **Weaknesses / Missing Content:** For readability, the abstract should focus on **key insights** rather than all results. Some detailed stats could be cut or phrased qualitatively. For example, instead of “model scores 45.5% on 3-round (p < 10^-7)”, one could say “achieves below-chance accuracy (≪50%)”. Also, the abstract assumes familiarity with concepts like “pp1” (percentage points) and VAE tests without introduction. A brief phrase on why these questions matter (one line motivating WHY we care about signal composition) might improve context. 

- **Unnecessary Content / Overlong:** The high-precision figures (99.99%, 97.78%, etc.) and multiple p-values make the abstract dense. They could be trimmed or moved to the main text. The parenthetical note “p < 10^-7” is more detail than needed in the abstract; it could be omitted or simplified to “statistically significant”.

- **Should be shortened / clarified:** The sentence “The signal resides in XOR-difference bits, is architecture-independent (≤2.6 pp across seven models), and persists to family-specific depths…” is already long and numeric. It could be split into two: one stating “The distinguishing signal resides in the XOR-difference channels and is essentially the same across different neural network architectures.” Another sentence can give a qualitative range of depths (e.g. “It remains detectable up to 9 rounds in Simon, 7 in Speck, 6 in Present” without exact pps). Similarly, the summary of results about “anti-transfer paradox” can be concise: “We formalize an ‘anti-transfer paradox’ where a model extracts almost all the information from shorter-round data yet predicts *wrong*, confirming the cipher itself is Markovian.”

- **Proposed Revision (excerpt):** 
  - *Original:* “The signal resides in XOR-difference bits, is architecture-independent (≤2.6 pp across seven models), and persists to family-specific depths (Simon32/64 9r ≫ Speck32/64 7r > Present-64/80 6r). SPN features compose hierarchically (≥96.9% positive transfer), but ARX and Feistel features anti-compose: ...”
  - *Rewrite:* “Our results show the neural distinguisher’s signal lies purely in XOR differences of cipher bits and does not depend on the network architecture. The signal persists through roughly 9 rounds for Simon, 7 for Speck, and 6 for Present. Moreover, while SPN-based ciphers exhibit conventional (positive) transfer of distinguishing features across rounds, ARX and Feistel ciphers exhibit a surprising **anti-transfer**: models trained on more rounds perform significantly below chance on fewer-round data. We formalize this **anti-transfer paradox**, and confirm (via mutual-information estimates and VAE-based tests) that it is not due to cipher non-Markovianity but is inherent to the learned neural representation.”

### Introduction

- **Purpose:** Set the stage: introduce context, importance, open problems, and summarize contributions.

- **Current Structure:**
  1. **Context & Motivation (Paragraph 1):** Mentions IoT, importance of cipher paradigms, question of vulnerability to attacks.
  2. **Classical vs. Neural (Paragraph 2):** References classical differential and the breakthrough of Gohr’s neural distinguisher.
  3. **State of Field:** References survey [11] on many papers, still open questions.
  4. **Open Questions (Representation & Compositionality):** Clearly lists the two main research questions to answer.
  5. **Contributions (Bullet-list 1–4):** Summarizes main findings and contributions to each question.
  6. (No explicit “Paper organization” or outline after contributions.)

- **Assessment of Purpose Achievement:** 
  - The introduction successfully identifies the big picture and poses clear questions. The contrast between classical and neural differential is well stated, highlighting why the new questions are needed. The contributions list is comprehensive, tying back to the questions.
  - However, the flow is choppy. There is no connecting “roadmap” sentence after contributions. The transition from contributions to Section 2 is abrupt. Also, no final paragraph outlines the structure (“this paper is organized as follows”).

- **Missing Content / Suggestions:** 
  - **Paper Organization:** Add a concluding paragraph to the intro that briefly outlines the rest of the paper. This is standard practice. E.g., “The rest of the paper is structured as follows: Section 2 surveys related work, Section 3 formalizes the problem, Section 4 describes our methodology, Section 5 presents experiments and findings, Section 6 discusses theoretical implications, and Section 7 concludes.”
  - **Clarification of Context:** Possibly more background on why transferability across rounds matters. For instance, classical theory would predict “positive transfer” under the Markov assumption; the intro hints at that but doesn’t explicitly say classical analysis suggests transferability, setting up why anti-transfer is unexpected. A sentence could be added: “Classically, differential features propagate under a Markov property, suggesting models should transfer positively from more rounds to fewer rounds.”
  - **Notational Clarifications:** The footnote about “pp” (percentage points) is referenced by a superscript ‘1’ as if a footnote, but isn’t clearly formatted. This should be handled either as a normal parenthetical remark (“accuracy differences of at most 2.6 percentage points”) or a proper footnote. Also, terms like DDT (difference distribution table) and MLP are used without definition – but they are established terms in the community, so this is minor.

- **Unnecessary/Too Detailed:** The contributions are very detailed. Some bullets contain multi-part results with numbers and significance. In an intro, it might be better to bullet only main qualitative points. Excess numeric detail (like “45.5% and 48.6% (p<0.05)”) could be trimmed or deferred to main text.

- **Flow Issues:** 
  - The “Contributions” appear as numbered bullet points. That’s okay, but bullet 4 (Markov–Transfer conjecture) starts immediately at page break with “Markov–Transfer conjecture”. On reading, it looked like part of contributions but its heading style is similar to section headings (“6 Anti-Transfer Paradox in Neural Cryptanalysis 3 / 4. Markov–Transfer conjecture”). The typesetting is confusing. Ensure that bullet 4 is clearly within the intro (it might have spilled onto page 3).
  - The phrase “Anti-Transfer Paradox in Neural Cryptanalysis 3” appears between contributions, likely a running head or page header, but could be mistaken for text by a reviewer. Review formatting.

- **Suggested Outline Revision:** The introduction might be reorganized slightly. One approach:
  1. First paragraph: Importance of neural distinguishers for lightweight ciphers. Briefly contrast with classical methods.
  2. Second paragraph: State the gap/open questions (signal location and compositionality under neural approach). This should mention classical expectation (“Markov assumption”) to motivate compositionality question.
  3. Third: “Contributions” bullet list – but consider reducing detail or splitting if too dense. Possibly move very detailed numeric results to later sections or footnotes.
  4. Fourth: “Organization of paper” paragraph as mentioned.

- **Section Splitting/Merging:** The Introduction is one section; it’s fine as is. Possibly split contributions into separate list or subsection (not needed as formal subsection, but can be a subparagraph).

### Related Work

- **Purpose:** Survey prior art to set context and differentiate this work.

- **Assessment:** The related work section is thorough and covers:
  - Classical differential (cite [8],[17],[18]).
  - Neural distinguishers (cite [12] and others, [6],[15],[14],[3],[22],[11]). It mentions specific model improvements and their scope.
  - Explainability papers ([7],[13],[1]) that directly relate to understanding neural features.
  - Cipher-specific improvements ([16],[20],[10]).
  - Transfer in neural crypto (basically points out the novelty: “to our knowledge, first empirical test of Markovian composability”).

- **Missing Content:**
  - **Preliminaries:** There's no separate preliminaries section, but in related work they touch on classical analysis and mention DDT. However, for completeness, a short “Preliminaries” might explicitly define standard terms (DDT, Markov property, differential propagation). Some conference papers include a preliminaries section. Here, related work partly serves that role for classical differential. If space allows, adding a brief “Preliminaries” or expanding the beginning of related work to explicitly define the Markovian assumption and DDT could help. E.g., “A block cipher is said to satisfy the differential Markov property if Pr[∆Cr | ∆Cr−1, …, ∆C0] = Pr[∆Cr | ∆Cr−1], meaning that given the input difference to the last round, earlier-round differences do not provide extra information.”
  - **Threat Model Clarification:** They do have "Threat model" in Problem Formulation (Sec 3). That's fine; I think they shouldn’t duplicate it here.
  - **Transfer Learning Literature:** The related work says “No prior work has measured transfer across rounds.” It might be worth briefly acknowledging any general literature on transfer learning in cryptanalysis (if any exists outside diff cryptanalysis) or say explicitly “beyond the context of cipher round-count transfer, transfer learning has been explored in other domains (cite?).” If none, that's fine but maybe be explicit: “We are not aware of any studies on transfer learning for cryptanalysis outside our work.”
  - **Reproducibility and Sources:** One could mention if any code or dataset from prior work is reused (probably not relevant here).

- **Unnecessary Content / Too Detailed:**
  - The paragraph on “Cipher-specific improvements” ([16],[20],[10]) seems least relevant to the core narrative. It says these works optimize for specific ciphers, and our contribution is cross-family. This could possibly be shortened. A one-sentence summary would suffice, rather than listing each work in detail, since the key point is just “others have focused on cipher-specific attacks, whereas we focus on cross-cipher analysis”.
  - The first paragraph on classical differential and linear is fine; it’s concise. 
  - The list of NN architectures (attention, gated units, SENet, etc.) shows breadth but might be overkill. The essential point is: “Many NN enhancements have been tried [6][15][14][3], as well as multi-pair inputs [22] and more ciphers [11].” Perhaps compress those into a shorter sentence or two, unless a particular one (like DBitNet [6]) is central.
  - Remove or tighten small asides like “published ResNet numbers” to focus on main content.

- **Flow / Transitions:** The section jumps through topics with very short paragraphs. It may read as bullet points in text. It might be clearer to group related items into paragraphs:
  - E.g. one paragraph on **classical vs neural** (the first two paragraphs already do that).
  - One paragraph on **analysis of neural-distinguisher behavior/interpretability** (the “Explainability” part).
  - One paragraph on **recent advances** (the cipher-specific attacks could fold into neural-diff work).
  - One paragraph on **transfer in neural cryptanalysis** (the last bit).
  
  This grouping would improve readability.

- **Placement Issues:** The reference to [11] (survey) in related work is good, but that same [11] is used as a survey of neural crypto. They correctly leverage it. The structure is logical: start broad (classical), then neural, then interpretability, then out-of-scope improvements, then gap identification.

- **Suggestions:** Possibly move “explainability” bullet into Methodology or Discussion context, but it does fit under related since it’s other work on understanding networks. I’d keep it here. Just ensure each subsection has a clear mini-topic.

### Problem Formulation (Section 3)

- **Purpose:** Define precisely the cryptographic setup, threat model, and what constitutes a distinguisher/advantage.

- **Assessment:** This section succinctly states the model: a chosen-plaintext attacker with random key, fixed input difference ∆P, outputs (C, C′), classification setting. It uses the terminology and references [6], [11] for context. It defines the distinguishing advantage ε and notes significance testing (one-sample t-test). This is good and clear.

- **Completeness:** It covers the essentials. It formalizes the classification task in line with Gohr [12]. It *does not* formally define DDT or other cryptanalytic notions here, but that is okay since it’s not needed beyond conceptual reference in related/discussion. 

- **Missing Content / Clarification:** Perhaps a sentence explicitly stating what it means for a neural distinguisher to “succeed”: e.g. “Our experiments measure how often a neural network correctly classifies real vs random ciphertext pairs; an accuracy significantly above 50% indicates a successful distinguisher.” This is implied but might be stated clearly. 

- **Unnecessary/Redundant:** The mention “All models use Adam…” belongs to methodology (it’s in sec4, not here). Problem Formulation stays focused on the task definition, which is good. No cuts needed.

- **Flow / Transition:** The section stands alone and does its job. No obvious improvements needed here. 

- **Suggestion:** Possibly add a sentence just after “We consider CPA with oracle access…” to connect the threat model to the goal: e.g. “The attacker’s goal is to distinguish real cipher outputs from random, given many plaintext pairs differing by ∆P.”

### Methodology (Section 4)

- **Purpose:** Describe how experiments are conducted: data generation, input representation, model architectures, baselines, and analysis techniques.

- **Subsections:**
  - 4.1 Data Generation and Training
  - 4.2 Input Representation
  - 4.3 Architecture
  - 4.4 Classical Baseline
  - 4.5 MINE (Mutual Info Estimation)
  - 4.6 Cross-Round Saliency
  - 4.7 Evolutionary ∆P Search

- **Assessment:** This section is comprehensive. It provides exact details on data sizes, training procedure, network setups, and even lists hyperparameters. It also covers the additional tools (MINE, saliency, genetic search) used to analyze representations.

- **Purpose Achievement:** It generally achieves its purpose of letting the reader know exactly how the experiments were done. It even mentions compute time (72 GPU-hours) in passing, which is thorough but perhaps more detail than needed in the main text.

- **Missing Content:** This is thorough, but a few possible clarifications:
  - In 4.1, they say “following Gohr’s protocol [12]” for generating 500k samples. That’s fine if Gohr’s paper is well-known, but maybe one brief phrase on what “balanced” means (equal real/random).
  - The notation “N = 500,000 balanced pairs” and “R2_xor_diff” are specific and technical; they eventually explain R2_xor_diff, but a high-level mention of what “R2” means (two rows? two ciphertexts?) could help. They do explain it in 4.2, so it’s okay.
  - In 4.4, classical baseline: It’s very brief and says “100k samples over 5 keys”. What about ∆P choice for baseline? Presumably same ∆P, but not stated. Maybe they mean “the same ∆P used for neural nets”. Clarity could be improved (e.g. “with the same ∆P and data”).
  - 4.5 mentions “MINE trains a statistics network T_θ to maximize a lower bound” with formula. One might wonder if this formula is too much detail for a cryptography paper, but it is standard for MINE. Still, it breaks the flow with math; maybe move Eq.(2) to a footnote or appendix? It’s borderline.
  
- **Unnecessary/Too Detailed:**
  - **Data Generation (4.1):** We might trim details like “5 seeds” (mention it generally) and maybe the exact early stopping patience (they say p<=5?), but not critical. The compute time statement (“72 GPU-hours”) can probably be omitted or moved to Appendix, as it’s an odd detail in the main story.
  - **Input Rep (4.2):** The table of dimensions “96 for 32-bit, 192 for Present” is fine. The detailed accuracy comparisons of 6 representations could perhaps be summarized. If space is tight, the parenthetical about R2_xor_diff being best could be shortened to just: “We adopt the “R2_xor_diff” encoding (including ciphertext bits and their XOR differences) since it gave near-optimal performance.” The exact “≤4pp of best” detail is nice but not essential. Possibly move the comparative detail to Appendix or omit.
  - **Architecture (4.3):** Listing seven architectures (MLP, CNN, Res. CNN, LSTM, GRU, Siamese, ResNet) and all their usage settings occupies space. Since the paper’s claims hinge on architecture-independence, it’s important to say “we tested multiple architectures (MLP, CNN, RNN, ResNet, etc.)”, but the full list might be trimmed or summarized in text rather than a table. Table 2 already shows 7 models’ results, so listing them here duplicates. We could cut: “We also evaluate six additional architectures (CNN, Residual CNN, Gohr-MLP, LSTM, GRU, Siamese) on Speck 5r, Simon 7r, Present 5r, plus a depth-10 ResNet on all ciphers.” If needed, put details (like exact sizes or references to models) in Appendix.
  - **Classical Baseline (4.4):** Fine as is but very short. It is probably okay, though they could merge 4.4 with methodology text or reduce if brevity needed.
  - **MINE (4.5):** The explanation “Crucially, MINE provides lower bounds: a low estimate does not prove low MI. We calibrate...” is useful. The equation (2) is perhaps too much detail in main text; it could maybe be relocated or omitted if brevity is needed. The main narrative can just say “We estimate I(H;Y) via MINE, comparing to random networks and same-round networks for calibration.” and push derivation to appendix. However, given this section is about methodology and not results, it may be acceptable to show the formula. 
  - **Saliency (4.6):** Fine, very brief.
  - **Evolutionary ∆P (4.7):** Also fine. One could question if detailing mutation rates and so on is necessary; maybe “We ran a genetic search over single-bit mutations to check other ∆P choices, with population 100 for 30 generations.” is enough.

- **Flow Issues / Improvements:**
  - The algorithm (Algorithm 1) appears in this section (likely under 4.x or 5). If it is meant to describe the cross-round evaluation procedure, it should be clearly placed (maybe at start of 4 or 5) with explanation. Currently it seems inserted awkwardly (our extraction showed it on page 4 top). It might actually be better as a numbered algorithm environment with a caption, either at the start of Methodology or at the end of Problem Formulation. If it’s just a narrative procedure, it could even be described in prose. Evaluate if the algorithm box is necessary or if a paragraph could suffice.
  - **Subsection merging:** The subsections are logically separate, but one could combine 4.1 and 4.2 (Data gen + Input rep) into one “Data and Input Encoding” subsection to save a heading line and shorten transitions. Similarly, the analysis tools (MINE, Saliency, ∆P search) are related to “how we probe the model”, so they could form a single subsection “4.x Analysis Tools” with sub-subsections for MI, saliency, ∆P search. But separate numbering is fine too for clarity.
  - The order is logical: data → architecture → analyses. One might consider moving the **Classical Baseline** either earlier (maybe right after data/training as part of experiment design) or later (perhaps to discussion as a mere reference). It’s so short it could be a paragraph instead of a full subsection.

- **Expanded / Too Short:** Possibly expand to clarify that the same ∆P is used for all tests and how keys are chosen or scheduled (they do mention key per seed). If multi-key effects or key-schedule differences matter (they test IRK later), that could be noted here or reserved for Discussion.

- **Recommendation:** Consolidate some subsections to save space. For example, merge 4.1/4.2 into one, or 4.5/4.6/4.7 into “Analysis Techniques”. Remove the formula or move it. Ensure the classical baseline rationale is clear but succinct. Remove minor details (like exact hyperparameter values, which could be Appendix). The goal is to make this section factually complete but lean in writing. 

### Experiments and Results (Section 5)

- **Purpose:** Present empirical findings, organized by theme or research question, and tie them back to the paper’s claims/questions.

- **Structure Identified:** The section is broken into subsections:
  5.1 Locating the Signal  
  5.2 Transfer Polarity as a Compositionality Test  
  5.3 The Anti-Transfer Paradox  
  5.4 Structural Controls  

- **Assessment:**
  - This section is data-rich and forms the core of the paper. It reports multiple tables and figures, and analysis corresponding to each research question.
  - **5.1 Locating the Signal:** Discusses Table 1 (accuracy vs rounds) and the resulting “signal depth hierarchy.” It concludes that Simon > Speck > Present in how many rounds are distinguishable. This addresses question about “where is signal” (in terms of how deep into rounds).
    - It also appears to incorporate discussion of architecture independence (Table 2 with multiple models) to show vulnerability ordering holds across architectures. That ties into “signal in structure, not architecture”. Good.
  - **5.2 Transfer Polarity:** Examines cross-round transfer in terms of “positive vs negative” for each cipher (likely based on Table 3 or text). Shows SPN yields positive transfer (~96.9% means almost always correct transfer), ARX/Feistel yield anti-transfer (accuracy below chance). Presents asymmetry (Speck→Simon vs Simon→Speck).
  - **5.3 Anti-Transfer Paradox:** Formal definition 1 is given here, quantifying conditions for anti-transfer. Provides example quantification of MI (MINE results showing high MI despite low accuracy). Introduces Fig.3 (accuracy vs MI plot) to illustrate the paradox. 
  - **5.4 Structural Controls:** This is a large set of sub-experiments (negated output, Markov validation via MINE, VAE, multiple ∆P, IRK, genetic search, ResNet). Figures 4 and 5 are included. This addresses potential alternate explanations and confirms cipher is Markovian.

- **Purpose Achievement:** The section thoroughly presents results tied to each question:
  - Location (5.1): Achieved.
  - Signal composition (5.2): Achieved.
  - Explanation of paradox (5.3, 5.4): Achieved.

- **Missing Content:** 
  - Possibly missing a concise statement of the null hypothesis or statistical validation plan. They report t-tests and Bonferroni in passing, but it might help to state clearly how significance is judged (though this was partly in Sec 3). For instance, “We consider results below 50% accuracy to be statistically below chance if p<0.05 after Bonferroni correction,” but they do mention this earlier in PF.
  - If possible, a small “experimental setup” preamble at the start of Section 5 listing common parameters (reuse of 500K samples, seeds, architecture, etc.) could orient the reader. They do put a sentence at top of Section 5, which is good.
  - For reproducibility, one might expect mention of software frameworks or hardware used. They gave “72 GPU-hours on a single GPU” in methodology — they could specify “trained on an NVIDIA [type]” etc. But this is not mandatory for acceptance.
  
- **Unnecessary / Move Elsewhere:** 
  - The table captions and entries occupy space. The tables themselves are essential, but maybe formatting can be optimized. For example, Table 1 is split across columns and spans pages (the extraction showed partial). Ensure that in final version, tables fit neatly and are clearly referred.
  - Table 4 (which likely summarized transfer polarity results) is referenced in text. If any tables are only numerical backup (like a long table of evolutionary ∆P results), those could be in Appendix.
  - The detailed description of control (iv) IRK includes a reference to Proposition 2 and [13] which is partly theoretical; it reads more like discussion. Perhaps some of the very detailed reasoning (“Proposition 2 resolves this...”) could go in Discussion or Appendix, with the main point (anti-transfer independent of IRK) given here.
  - The sequence of 5.4 (i) through (vi) is long. It might be better to label these as (i)–(vi) as done, but also summarize each with a one-liner conclusion before the evidence. For example, “(i) Negated-output: flips fail to restore accuracy (54.5% vs ~99.99%), indicating features differ structurally, not just sign.” Currently it mixes explanation and result in one long paragraph. Breaking it up could improve clarity. However, that’s writing style; content-wise it's fine.

- **Expanding / Shortening:** 
  - Sections 5.1–5.2 are succinct. Section 5.3’s definition is needed (though it’s a bit formulaic). Possibly shorten the narrative around the definition (some of (i),(ii) can be bulletized or simplified).
  - Section 5.4 is very long. It might be trimmed by moving some details to an Appendix titled “Additional Experimental Controls.” The main text can then say “We performed several control experiments, summarized here, to rule out artifacts.” and list the key findings (negation, Markov test, ∆P invariance, IRK, ∆P search, ResNet replication). Full descriptions of each (especially Markov property test with VAEs) might be partly in Appendix. Figures 4 and 5 should remain, but some of the text describing them (especially low-level detail of VAE MSE ratios) could be shortened.

- **Suggestions for Section Splitting / Merging:** 
  - The subsections are logical and tied to research questions, which is good. Possibly **5.4 Structural Controls** could be broken into two subsubsections: one for “Testing Cipher Markovianity (Markov Validation, VAE)” and one for “Additional Controls (negation, ∆P, IRK, ResNet)”. But since 5.4 is already a main subsection, adding more layers might be overkill.
  - It might be clearer to rename “Structural Controls” to something like “Validation and Robustness Checks” so readers understand these are sanity checks rather than new results.

- **Material Movement:** 
  - The actual data from evolutionary search, IRK differences, etc., if very detailed (like exact percentages for each ∆P or each seed), can be in Appendix. In main text, summarizing the outcome is enough.
  - The formal **Definition 1** might arguably belong in the discussion section (theory) rather than experimental results. However, they use it to “quantify” the observed paradox. It’s a bit unusual: either treat it as a main concept early (introduction/discussion) or keep as is. It’s not too problematic here but can be noted.
  
- **Revised Outline for Section 5:** One could consider merging 5.3 and 5.4 under a common theme “Anti-Transfer Findings” with subsections, but current numbering is okay. The outline might remain:

    - 5.1 Locating the Signal (Table 1, 2)
    - 5.2 Transfer Polarity (Table for transfer accuracies)
    - 5.3 Anti-Transfer Paradox (Definition, MINE, Fig 3)
    - 5.4 Robustness Controls (i–vi bullets, Figs 4–5)

### Discussion (Section 6)

- **Purpose:** Interpret results, present formal analysis, explain underlying causes, compare to theory, acknowledge limitations.

- **Subsections in Discussion:**
  6.1 Markov–Transfer Theorem (formal)
  6.2 Mechanistic Analysis for Simon (AND-gate lemma etc)
  6.3 Limitations

- **Assessment:** This is the theoretical core. It’s unusual for a crypto paper to have both empirical results and a formal theorem in one paper, but here it’s a strength. 

  - **6.1 Formal Theorem:** Theorem 1 is clearly stated with conditions C1–C3, then the *key idea* and a Proposition (in Appendix). It’s well structured. The subsequent text explains the significance and relates to SPN (Proposition 1) in Appendix. This section is dense but well-justified: it ties the empirical observations to cryptographic theory.
  - **6.2 Mechanistic Explanation:** The breakdown of why Simon (a Feistel) behaves differently is good. Lemma 1 is stated and proved (simple XOR/AND algebra), then (i),(ii),(iii) listed, then empirical evidence. This reads slightly like a mini-paper within the paper, but is relevant because it explains the anti-transfer. It’s a bit long, but cryptographers will appreciate the rigor.
  - **6.3 Limitations:** It’s great they included this. It frankly addresses points a reviewer might raise (model capacity, multi-pair vs single, baseline, block size). This shows authors thought about criticisms. It also notes that none of these would invalidate the main phenomenon.

- **Purpose Achievement:**  
  - This section strongly accomplishes “explain and contextualize.” It presents a new theorem (the authors even call it a conjecture-turned-theorem), which is unusual but compelling. It then validates the cause of anti-transfer with theory and data. It fully addresses “why” after “what” was shown.
  - It also includes an honest limitations discussion, which is often missing in conference papers. This is a positive.

- **Missing Content:** 
  - Possibly a few items:
    - A small “Future Work” or “Open Questions” part might fit here or at conclusion, although they started listing that in the conclusion. They have some future directions in the tail end of conclusion (extending theorem to Feistel, etc.).
    - They mention figures 4 and 5 in 5.4, but here in discussion they use terms from Section 5 (“Fig. 4 shows…Fig. 5 provides…”). Actually those are in Section 5. They probably should say “as shown in Fig. 4 (Section 5)” etc. Make sure references to figures in discussion are correct (they might be reversed due to extraction; the figure captions we saw were for 4 and 5 in section 5).
    - Theorem 1 is very useful; perhaps they could add a short corollary in text form summarizing it in plain language (if it’s not too redundant). E.g. “In essence, if the cipher is truly Markovian and the distinguisher only uses classical differential features with monotonic bias decay, then any model’s advantage should *not* worsen on smaller-round data. This formalizes the classical expectation of positive transfer.” That might help readers grasp it without wading through conditions.

- **Unnecessary/Move:** 
  - Section 6 is mostly essential. The detailed derivation parts could, in principle, be appendix. They have already moved some to Appendix A (propositions). If space is an issue, some of the text in 6.2 (the bullet list (i)–(iii) plus Lemma) could be truncated, saying “Key properties of Simon’s round function (data-dependent AND, half-state update, rotations) cause bias sign flips. We verify this empirically.” However, since this is the novel insight, it’s probably worth keeping most of it. One could argue the explicit proof of Lemma 1 is almost textbook, but including it is fine.
  - The limitations subsection might be a bit too detailed for some readers, but it’s useful. Could consider shortening “(iii) baseline” since classical results [13] already mentioned, but it’s one line, so okay.

- **Flow / Transitions:**  
  - 6.1 to 6.2: There is a clear conceptual break (“From Markov to Transfer” vs “Why Simon anti-transfers”). 6.2 starts with “Gohr et al. [13] proved Feistel only learns DDT features, so anti-transfer must come from violation of (C3).” This connects well as it references (C3) from the theorem. Transitions within 6.2 (into Lemma) are reasonable.
  - The bullet list in 6.2 is maybe too indented in our text view; ensure formatting is clear (they used roman numerals).
  - 6.3 (Limitations) starts with a numbering “1.” which is weird as it is a new subsection. Possibly they meant a numbered list of limitations; that’s okay as prose style.

- **Section Naming:** 
  - “Discussion” as a heading is generic; the subsections are fine. Some papers rename “Discussion” to “Analysis” or “Theory and Discussion” when it includes formal results. It might even be split into “Theory” (6.1–6.2) and “Discussion/Limitations” (6.3), but this is not crucial.

- **Additions:** The authors could consider a brief “Related Cryptanalytic Context” tie-in, but they already did theory. Possibly mention the real impact on key recovery: they hint at this in conclusion (“transfer-based key recovery requires composable signal”). If not already clear, maybe say explicitly in Limitations or a final paragraph, “Thus, an attacker cannot simply train on many rounds and apply to fewer; new strategies are needed for ARX/Feistel.”

### Conclusion (Section 7)

- **Purpose:** Summarize the key findings, answer the big questions, and state implications and future work.

- **Assessment:** The conclusion effectively summarizes the contributions as restated bullets: representation (signal location and architecture-independence), compositionality (SPN vs ARX/Feistel), formalization (Theorem and Lemma references), empirical confirmation (MI, VAE). It also gives practical implications for key recovery and cipher design, and a brief note on future directions.

- **Missing/Can Add:** 
  - They have a short “Future directions” at the end (i and presumably ii). The snippet ends at “(i) extending theorem;” presumably (ii) is in continuation. This is good and should be completed. Possibly explicitly state (ii) if truncated: maybe “(ii) Applying this understanding to other cipher families or larger block sizes, or exploring how multi-pair distinguishers behave.”
  - The practical implications could be a bit clearer on impact (they mention “cipher designers” and key recovery).
  - A one-sentence “final takeaway” might help (“In sum, we show that neural distinguishers do obey classical Markov diffusion for SPN but can actively mislead for ARX/Feistel, a surprising and important phenomenon for future cryptanalysis.”).

- **Flow:** The conclusion reads like a compact summary, which is good. No new details are introduced (except maybe “Future directions”). 

- **Add Outline Reference:** If not done earlier, it’s fine to not outline in conclusion as long as intro had organization. But if intro lacked it, some do it in conclusion too. Not needed if done in intro.

- **Eliminate Repetition:** It largely repeats contributions (which is fine) but maybe doesn’t need to restate all numbers (they do keep some, but it’s okay to have them again succinctly).
  
- **Formatting:** The line break “Anti-Transfer Paradox in Neural Cryptanalysis 15” suggests the page header repeated. Ensure that’s not in final text.

- **Page-limit Concern:** Conclusion is already moderate length; if rewriting to be more concise, maybe trim repetition. But it’s okay as is.

## Revised Paper Outline Proposal

Given the above analysis, I propose the following revised outline (names and hierarchy):

1. **Introduction (≈1–2 pages)**  
   - Motivation and background (lightweight block ciphers, neural vs classical)  
   - Open questions (Representation, Compositionality)  
   - Main contributions (concise bullet points, reduced numerical detail)  
   - Organization of paper  

2. **Related Work (≈1.5 pages)**  
   - Differential cryptanalysis and classical Markov property (with brief definitions)  
   - Neural differential distinguishers (Gohr [12] and follow-up)  
   - Explainable/interpretable cryptanalysis (Benamira [7], Gohr [13])  
   - (Brief) other neural cryptanalysis advances (multi-cipher, architectures)  
   - Gap: transfer learning and composability in neural cryptanalysis  

3. **Problem Formulation (≈0.5–0.75 pages)**  
   - Threat model (CPA, chosen ∆P, oracle, classification task)  
   - Definition of distinguisher and advantage (ε)  
   - Criteria for success (accuracy and significance)  

4. **Methodology (≈2.5 pages)**  
   - Data generation and encoding (explain Gohr’s data gen in one paragraph)  
   - Input representation (bitwise and XOR differences)  
   - Neural network architectures (describe main MLP; list others tested)  
   - Classical baseline (simple single-bit distinguisher)  
   - Analysis tools:  
     - MINE for MI estimation (equation optional or footnote)  
     - SmoothGrad saliency (how used)  
     - Genetic ∆P search (brief description)  

5. **Experiments and Results (≈6–7 pages)**  
   - **5.1 Signal Localization:** Table of accuracies vs rounds; depth hierarchy; architecture comparison (Table) and discussion (Figure for ordering if any).  
   - **5.2 Compositionality (Transfer Polarity):** Table/figure for transfer results; interpret positive vs anti-transfer; cross-cipher asymmetry.  
   - **5.3 Anti-Transfer Paradox:** Formal definition, example quantification (MINE vs accuracy), Fig. 3 (accuracy vs MI), explanation of paradox.  
   - **5.4 Robustness Controls:** Summary of control experiments (i–vi). Key findings bulletized (or small paragraphs with bold labels). Figures 4–5 (VAE/saliency) as needed.  

6. **Discussion / Theoretical Analysis (≈4 pages)**  
   - **6.1 Markov–Transfer Theorem:** State Theorem 1 clearly with conditions and conclusion. Intuition and connection to our results (monotonic bias). (Proof sketch or move to Appendix.)  
   - **6.2 Mechanism for Feistel Anti-Transfer:** Lemma on XOR&AND propagation; explanation of Feistel-specific effects; empirical bias-sign-flip evidence.  
   - **6.3 Implications and Limitations:**  
     - Summarize implications for cryptanalysis (e.g. multi-round trained models often fail on fewer rounds for ARX/Feistel).  
     - Limitations (architecture, single-pair vs multi-pair, baseline approach, block size).  
     - Possibly future work here or next section.  

7. **Conclusion (≈0.5–1 page)**  
   - Recap of questions and answers (signal location, compositionality findings)  
   - Theorem and lemma results briefly restated  
   - Practical takeaways (key recovery, design insight)  
   - Future directions (as bullet or list)  

**Appendix (up to 10 pages)**  
   - Formal proofs of Theorem 1 and Lemma 1 (if not fully in main text)  
   - Extra experimental details (hyperparameters, additional tables, raw data for controls)  
   - Possibly extended discussion of related theoretical points (like details of SPN proof) if needed.  

**Notes on Outline Adjustments:**  
- I’ve merged some sections (like combining all analysis tools under Methodology rather than separate 4.x items) to reduce section headings.  
- “Discussion” is renamed to a combined “Discussion / Theoretical Analysis” section to reflect both empirical discussion and formal theory.  
- The core narrative flow is: *Setup → Results (by theme) → Theory & explanation → Wrap-up.*  
- Redundant or overly detailed pieces (like listing every variant architecture or cipher-specific improvement) are downplayed to emphasize the main story.  

The aim is to present the same content more linearly: first present **what** we found about the signal (Sec.5.1–5.2), then **define and quantify** the paradox (Sec.5.3), then **validate** it (Sec.5.4). After that, Sec.6 steps back to **explain why** these phenomena occur (theorem and mechanism).

## Part 3: Missing Conference-Paper Components

Comparing to typical strong CANS/ACNS/ToSC/CRYPTO papers, here are components that are missing or could be made more explicit:

- **1. Preliminaries / Notation:** Reviewers often expect a short “Preliminaries” if any specialized notation or concept is used. Here, the draft *implicitly* uses terms like DDT (difference distribution table), Markov property, and Walsh-Hadamard transforms without defining them. While experts know these, adding a brief prelim section (or an expanded part of Related Work) defining key terms like “differential distribution table” and “Markov propagation” would help clarity. *Why expected:* It ensures all readers understand the assumptions (especially the Markov one) and notation. *Impact:* Medium. Without it, some reviewers might feel lost or that the paper assumes too much. *Placement:* Insert after Related Work, before Problem Formulation (0.25–0.5 page).

- **2. Threat Model Clarification (if needed):** The threat model is given, but a one-line clarification of attacker goals (e.g., “The adversary’s goal is to recover key bits or distinguish real from random ciphertexts with advantage ε.”) could make explicit why distinguishing accuracy matters. *Hurt if missing:* Low, since they do cover it, but explicitness is always safer. *Placement:* End of Sec. 3 or in Problem Formulation.

- **3. Formal Definitions:** The paper introduces a *Definition 1* for the anti-transfer paradox, which is good. However, it does not formally define some other things, e.g., “transfer polarity” or “compose hierarchically.” If these terms appear in results, they should be defined or at least used consistently. For example, “positive transfer” and “anti-transfer” could each have a brief definition or descriptive explanation when first used (they do describe it in text, but maybe a one-line formal def would help). *Impact:* Low to medium – clarity helps reviewers.

- **4. Running Example or Intuition:** While not common in crypto papers, having a small illustrative example (maybe a toy cipher or a figure) could sometimes help. For instance, a diagram illustrating how XOR differences propagate vs. AND differences could guide intuition. However, given page constraints, this is optional. Without it, the paper may feel too abstract at first. *Impact:* Low, space cost maybe 0.5 page for a simple figure and caption. But not crucial.

- **5. Attack Intuition Section:** Often cryptography papers have a short “Attack Overview” or “High-Level Idea” after introduction, especially if the details are complex. Here, the narrative is somewhat linear (the answers themselves are the “attacks”). It might help to add a short overview (maybe at the end of Intro) describing the key experiments and logic flow: e.g. “We first locate the signal via single-round tests, then assess cross-round transfer with trained networks, then analyze the paradox through mutual information and symbolic proofs.” This orients the reader. *Impact:* Medium; it can improve readability.

- **6. Experimental Setup Details:** The methodology is detailed, but a specialized subsection listing hardware (GPU type, OS, random seed management, libraries used) is missing. Modern conference papers often have a sentence “We implemented our neural networks in PyTorch/TensorFlow and ran experiments on an NVIDIA GPU X.” This helps reproducibility. *Impact:* Low-medium, but reviewers sometimes note missing these details. *Placement:* At end of Methodology, or in a paragraph labelled “Experimental Setup”.

- **7. Reproducibility Note:** No explicit mention of releasing code/data. Adding a line like “Code and trained models will be made public upon publication” or “Upon acceptance, we will release our code” is helpful. *Impact:* Low, but positive for reviewers.

- **8. Limitations Section:** The draft does have a “Limitations” paragraph (6.3), which is often appreciated but not always present. So they have this – good. Perhaps it should be more clearly titled (e.g., “Limitations and Future Work”) and maybe include “Future Work” as separate points. Right now future work is only briefly mentioned in conclusion.

- **9. Theorem Statements:** They have Theorem 1 and Lemma 1, which is excellent. It might be good to number the definition of Anti-Transfer (Definition 1) and refer to it by number in text when needed for clarity. They already do this, so that’s fine. Just ensure all formal statements (Theorem 1, Lemma 1) are clearly numbered and referenced.

- **10. Statistical Validation Section:** They report t-tests and confidences throughout. However, explicitly saying how many trials and how confidence intervals are computed can be explicit. They do mention 5 seeds and bootstrapped CIs. Perhaps a short “Statistical methods” mention in methodology or experiments stating “We report mean±std over 5 seeds and use bootstrapped 95% confidence intervals for boundary cases (<55% accuracy). All hypothesis tests use one-sample t-tests at α=0.05 (Bonferroni-corrected where appropriate).” *Impact:* Medium. Clear statistical rigor is expected.

- **11. Practical Implications Section:** They cover this a bit in conclusion (key recovery and design). Perhaps reviewers would like a more explicit “Implications” subheading in discussion or conclusion linking back to application. For instance, “Key recovery via transfer relies on positive transfer; our results imply that such techniques will fail for ARX/Feistel beyond a small gap. Designers might favor SPN for this reason.” This is touched on, but highlighting it as its own point can strengthen the “broader impact” part. *Impact:* Medium. It rounds off why we care.

- **12. Citations for Key Claims:** There are a few places where reviewers might ask for citations:
  - The Markov property of ciphers is mentioned. They could cite a source or state it's known folklore (though presumably they proved it with VAE). It is fine as is.
  - The claim “no prior work on cross-round transfer”: they cite [2] to indicate one related attempt; maybe also cite [13] and [11] more explicitly here. But mostly the onus is to show novelty, which they do.
  - For “Sharp improvements to some ciphers via neural nets” maybe not needed, since focus is on this work.
  
- **13. Running Headings, Section Titles:** As noted, “Algorithm 1” and random headers (“Anonymous”) should be fixed. Ensure all headings and figure/table captions are clear. This is minor but can trip formatting reviewers.

Each missing component’s *acceptance impact*:
- Preliminary definitions / threat model clarity: Medium (improves understandability for all PC members). 
- Formal defs (DDT, Markov): Medium.
- Outline / intuition: Medium to High (lack of organization hurts reading).
- Reproducibility details: Low to Medium.
- Practical implications emphasized: Low to Medium.
- Statistical procedure clarity: Medium.
- Limitations already present: High (good thing they have it).
- Code release mention: Low but positive.

**Where to Insert:** 
- Preliminaries: after related or as part of intro if short.
- Attack overview: at end of Intro.
- Stats details: at end of Methodology or Experiments intro.
- Code release: maybe in footnote on first mention of experiments or conclusion.
- Implications: separate paragraph in Conclusion (they have some of this).

## Part 4: Writing Quality Audit

Below are specific issues found in the text, with suggestions and rewrites. Page/line references are approximate given formatting:

- **Overly long paragraph (Intro, 1st paragraph):** The opening paragraph runs on (~6 lines) before breaking. It covers IoT, cipher paradigms, and question of vulnerability. Suggest breaking into two shorter paragraphs: one on context (lightweight ciphers and why their security matters), second on classical vs neural differential (or vice versa).
  - *Example rewrite:*  
    - *Original (intro):* “Lightweight block ciphers—built on ARX, Feistel, or SPN paradigms—protect billions of IoT sensors…direct implications for cipher selection.”  
    - *Rewrite:*  
      “Lightweight block ciphers—built on ARX (Add-Rotate-XOR), Feistel, or Substitution-Permutation (SPN) structures—protect billions of IoT devices. A natural question is whether one design paradigm is inherently more vulnerable to a given attack. Classical differential cryptanalysis [8] would answer this by searching for differential trails, but this approach becomes infeasible at higher rounds. Recently, neural network-based distinguishers have broken these limits: Gohr [12] used a 10-layer ResNet to distinguish 8-round Speck32/64. This raises new questions about *how* these neural distinguishers work, beyond just *if* they work.”
    (This splits context and prior work, adds connective “raises questions.”)

- **Informal style (“we answer both...”, “we address two foundational questions”):** Using “we” repeatedly is acceptable, but some phrases are too informal or hype-like. For example, “two foundational questions” can sound grandiose. Consider more measured language: “two key open questions” or “two fundamental questions” if justified.
  - *Rewrite example:* Instead of *“two foundational questions remain unanswered”*, say *“two key questions remain open”* or *“two important questions are yet unresolved.”*

- **Footnote formatting (“pp1” glitch):** In the abstract and intro, the text shows “pp1” (as if a footnote marker). This likely meant “percentage points (pp)”. Fix by writing it out or with a footnote properly. E.g. change “≤2.6 pp1” to “at most 2.6 percentage points”. Remove the stray “1”.
  
- **List of contributions (Intro):** The numbered bullets are a bit dense. For instance, item 1 is a single very long sentence with semicolons. It could be split for clarity:
  - *Original:* “The vulnerability ordering Simon32/64 ≫ Speck32/64 > Present-64/80 holds across seven architectures (≤2 pp spread) and under Gohr's ResNet (≤2.6 pp gap), establishing that the signal resides in cipher structure, not model capacity (Sect. 5.1).”  
  - *Rewrite:* “**1. Architecture-independent signal.** Across all architectures we tried (MLP, CNN, ResNet, etc.), the round-depth vulnerability ordering *Simon32/64 > Speck32/64 > Present-64/80* remains intact (accuracy differences ≤2 percentage points). Even Gohr’s ResNet has only a 2.6 pp difference from our MLP. This shows the distinguishing signal is due to the cipher itself, not the network’s capacity (Section 5.1).”
  This is more itemized and splits the content into shorter clauses.
  
- **Transitional phrases:** Some jumps are abrupt. For example, the end of Introduction has a marker “Anti-Transfer Paradox in Neural Cryptanalysis 3 4. Markov–Transfer conjecture.” This looks like a page header leak. More generally, at the end of Section 5.3, they say “(Crucially, while MINE serves as an initial probe, we definitively validate the underlying Markovian assumption using exact conditional MI and VAE tests in Sect. 5.4, confirming these signals point to a real structural phenomenon rather than an estimation artifact.)” That parenthetical is quite long and a bit buried. It could be a separate sentence. 
  - *Rewrite suggestion:*  
    “Importantly, these MINE results are only preliminary: we will validate the cipher’s Markov property more rigorously using exact conditional mutual information and VAE-based tests (Section 5.4). The results confirm that the anti-transfer is a genuine structural phenomenon, not an artifact of estimation.”

- **Repetition:** Phrases like “architecture-independent” appear multiple times. It’s a key point, so some repetition is okay, but try to vary. Similarly, “anti-transfer” is central, but ensure every occurrence is clear. There are instances of writing “Speck32/64 5-round model scores 45.5%” and elsewhere “45.5%” too – consistent decimal vs digits is fine, but be uniform (“45.5%” vs “45.5 %” spacing, etc.).

- **Redundant explanations:** In Methodology 4.5 they note “MINE provides lower bounds: a low estimate does not prove low MI.” This is a very useful point, but it’s slightly repetitive to state again “lower bound, so cannot prove low MI” when they then speak of calibration. It's fine as is, but could be tightened: “Since MINE only gives a lower bound on MI, we calibrate each estimate against a random (noise-floor) network and a same-round network (upper reference).”

- **Weak transitions:** Occasionally sections jump without a lead-in. For instance, at the start of 5.2 they abruptly refer to “Table 4” without first summarizing what Table 4 is. It would be smoother to start the paragraph by stating what is being tested, then “As Table 4 shows...”.
  
- **Informal language:** Phrases like “We now formalize...” or “Crucially, ...” are okay in moderation but could be more formal. Use “We formalize and prove...” only when really needed. “Notably” or “Importantly” can replace “Crucially”.

- **Grammar/typos:** Check for minor OCR or formatting errors: e.g. “(≥96.9% positive transfer)” – is the bracket spacing normal? The extraction had some gibberish like “(5.35.4)” which likely should be “(Sections 5.3–5.4)”. Make sure all ranges and citations have correct symbols. Also “eect” should be “effect” in text (OCR artifact).

- **Blog-post tone:** Overall, the writing is mostly formal. There are no obvious first-person anecdotes or overly casual asides. The main corrections are splitting long sentences, clarifying jargon, and tightening phrasing.

- **Example Rewrites:**

  1. *Problem Formulation lead-in:*  
     - *Original:* “Threat model. We consider a chosen-plaintext attacker (CPA) with oracle access to an r-round block cipher E^r_K keyed by a uniformly random secret key K.”  
     - *Rewrite:* “**Threat Model.** We assume a chosen-plaintext attack (CPA) setting: the adversary queries pairs (P, P⊕∆P) and obtains ciphertexts under an unknown random key K of an r-round cipher E^r_K. The attacker’s goal is to distinguish such ciphertext pairs from random.”
  
  2. *Methodology 4.1 first sentence:*  
     - *Original:* “For each cipher at r rounds, we generate N = 500,000 balanced ciphertext pairs following Gohr’s protocol [12]: positive samples encrypt (P, P ⊕∆P) under a random key; negative samples encrypt two independent plaintexts under the same key.”  
     - *Rewrite:* “For each cipher and round count r, we follow Gohr’s procedure [12] to generate N=500,000 balanced ciphertext pairs. Positive samples are (P, P⊕∆P) encrypted under one random key, while negative samples consist of two independent plaintexts encrypted under the same key. This yields equal numbers of “real” and “random” examples for training.”

  3. *5.1 Locating the Signal lead-in:*  
     - *Original:* “Signal depth. Table 1 reveals a clear signal depth hierarchy. Simon32/64 retains distinguishable signal through 9 rounds… Speck… through 7r, Present… through 6r.”  
     - *Rewrite:* “**Signal Depth.** Table 1 shows the classifier accuracy versus the number of rounds for each cipher family. We observe a clear hierarchy: Simon32/64 yields accuracies above chance up to 9 rounds, Speck32/64 up to 7, and Present-64/80 up to 6. In other words, the neural distinguishing signal persists for more rounds in Simon (Feistel) than in Speck (ARX), and persists the least in Present (SPN).” (Then comment on diffusion and this ordering.)

  4. *Anti-Transfer definition (5.3):*  
     - *Original:* “Definition 1 (Anti-Transfer Paradox). Let D_r denote the data distribution at round r… Write acc(f^(r), D_r') for… and I^(h(r);Y | D_r') for... The anti-transfer paradox at (r,r') occurs when: (i) acc(f^(r),D_r') < 1/2 – δ ... (ii) I^(h(r);Y | D_r') ≥ γ · I^(h(r');Y | D_r') …”  
     - *Rewrite:* This is formal and fine, but could use a brief introductory line: “We now define the *anti-transfer paradox* formally. Let f^(r) be a distinguisher trained on r-round data with penultimate-layer representation h^(r). For a target round r'<r, denote by acc(f^(r), D_{r'}) the accuracy of f^(r) on r'-round data and by \widehat{I}(h^(r);Y | D_{r'}) the estimated mutual information between h^(r) and the label Y at round r'. Then:  
       - (i) **Below-chance accuracy:** acc(f^(r), D_{r'}) < 0.5 – δ (significantly below 50%), and  
       - (ii) **High relative MI:** \widehat{I}(h^(r);Y | D_{r'}) ≥ γ · \widehat{I}(h^(r');Y | D_{r'}), where h^(r') is a model trained on r' rounds.  

       If both conditions hold (with δ,γ chosen suitably), we say an anti-transfer paradox occurs at (r→r').”  
     This clarifies the logic for the reader without altering the math.

  5. *Conclusion start:*  
     - *Original:* “We address two foundational questions in neural differential cryptanalysis…”  
     - *Rewrite:* “We have addressed two key questions in neural differential cryptanalysis via a controlled study on Speck32/64, Simon32/64, and Present-64/80. **Representation:** The distinguishing signal is localized entirely in XOR differences of ciphertext bits, and we find that this signal is consistent across different neural architectures (accuracy varies by ≤2 percentage points). The signal endures to a cipher-specific depth (about 9 rounds for Simon, 7 for Speck, 6 for Present). **Compositionality:** SPN-based ciphers (Present) exhibit conventional positive transfer (trained-on-7r model still classifies 6r data correctly ~99% of the time), whereas ARX/Feistel ciphers show anti-transfer: for example, a Speck32/64 5r model achieves only ~45% accuracy on 3r data (significantly below chance).”  
     (Continue similarly for the rest.)

Overall, rewriting should focus on simplifying sentences, breaking up overly complex ideas, and making implicit logical connections explicit. The tone should remain formal and concise.

## Part 5: Storytelling and Narrative

**Central Question:** The core scientific questions are (1) *Signal Localization*: *Which bit-level differences carry the neural distinguisher’s signal, and how does this depend on cipher and model?* and (2) *Signal Composition*: *Does the distinguishing signal at r rounds compose (in the Markov sense) to r′<r, allowing transfer from deeper models to shallower data?* 

This should be obvious early: the introduction already states them, but we should emphasize why these questions matter. Classical diff cryptanalysis expects a Markov property (signals compose), so questioning that in neural nets is a strong hook.

**First Page Narrative:** 
- The reader should quickly learn “Neural distinguishers outperform classical ones, but we don’t know *how* they find their signals and whether classical intuition still applies.” 
- The first page should pose the problem: “Neural net can learn beyond known differential trails; can we pinpoint where it looks? And does a model trained on more rounds still work on fewer rounds, or does something break?”
- Ideally, the intro ends with a clear statement: *“This paper finds that, surprisingly, neural distinguishers do *not* obey the classical Markov composition in many cases. We document a new phenomenon (‘anti-transfer’) where deeper models perform below chance on fewer rounds.”* This will hook the reader.

**Section-by-Section Story Arc (redesigned):**

1. **Introduction:** 
   - Educate on context → (Perhaps mention Fig.1 overview now, so reader sees a cartoon of transfer vs anti-transfer.)
   - Raise questions: “What features do neural nets use (XOR-diff bits?), and do these features combine across rounds as theory predicts?”
   - Main results in simple terms: “We discover SPN ciphers behave as expected (positive transfer), but ARX/Feistel do *opposite* – we call this anti-transfer. We prove when positive transfer should hold (Theorem) and show why Feistel violates it (AND gate lemma).”
   - Raise expectation: “This has implications for neural attack strategies: transferring a model to fewer rounds can be disastrous for certain ciphers.”

2. **Background (Related Work):** 
   - Reminds reader of classical theory and known neural results.
   - Sets expectation: classical Markov propagation would suggest models should transfer. Prior works haven’t tested this.

3. **Methods / Setup:**
   - The reader should know the experimental setup: what ciphers, what task, what data is used. This should reassure them the experiments are fair.
   - Quick snapshot: “We train simple MLPs (and others) on r-round data, then test on r′-round. We measure accuracy and also analyze learned features via saliency and MI.”

4. **Results Part 1 – Signal and Baseline:** 
   - Present Table of accuracy vs rounds (Sec5.1). Learnable signal persists to certain depths. That tells reader “XOR differences carry a signal up to depth D.”
   - Present architecture comparison (maybe briefly in text) to say “Different nets see the same depth, so it’s cipher property.” 
   - *Question raised:* “What bits exactly? The findings above suggest high-bit XOR differences, not ARX-specific linear combos, since it’s architecture-agnostic.”

5. **Results Part 2 – Transfer Tests:** 
   - Now raise the transfer question explicitly: “Given a model trained on depth r, does it succeed on r′? Classical Markov says yes.” 
   - Show results: SPN ciphers are fine (almost 100% positive transfer); ARX/Feistel are *bad* (accuracy ≈45–49%).
   - Show asymmetry (Speck→Simon anti-transfer, not vice versa). 
   - The reader now learns: “This is surprising – the model is *actively misclassifying* instead of carrying over the trend.”

6. **Results Part 3 – Paradox Formalized:** 
   - Define the anti-transfer paradox succinctly: high retained mutual information *but* wrong class. 
   - Refer to MINE/MI analysis: the model’s representation still contains *information* about the input differences (via MI), yet its output labels are flipped.
   - Fig.3 puts MI vs accuracy; highlight that for ARX/Feistel, MI stays high while accuracy plummets below chance, confirming the paradox.  
   - *Question now:* Why does this happen? Is the cipher at fault, or is it the network?

7. **Results Part 4 – Controls/Markov Tests:** 
   - To confirm it’s not the cipher, do controls: e.g., measure cipher’s Markov property directly (via conditional MI or VAE). 
   - Show that the cipher IS memoryless (Fig.5 VAE overlapping). 
   - Other controls (negated output, different ∆P, IRK test) all point to the same: the phenomenon is due to the learned model representation, not some bug in data.
   - At this point the reader sees: we have strong evidence and controls that this anti-transfer is real and not a fluke.

8. **Discussion / Theory – The Big Explanation:** 
   - Now zoom out: Classical theory said positive transfer should hold under Markov and DDT-limited features. We have that for SPN (Proposition 1 yields monotonic bias).
   - State and explain Theorem 1: *If the cipher is Markov and the model uses only classical differential features with monotonic bias, then positive transfer is guaranteed.* We show SPN satisfies conditions, consistent with experiment.
   - Conversely, for ARX/Feistel: show which condition fails. Give Lemma 1 and the three points (i–iii). Conclude: Feistel’s AND gates cause bias sign flips, violating monotonicity. This *theoretically* explains anti-transfer. Cite the evidence (bias flips, negative adjacent correlations).
   - Draw the reader’s attention: this has not been observed before in cryptanalysis literature; it’s a novel insight.

9. **Conclusion:** 
   - Summarize: “Yes, we found the answers to the questions. The signal is in XOR bits. Compositionality holds for SPN but fails for ARX/Feistel. We proved why. 
   - Emphasize what the reader should now understand: “Neural nets can exhibit surprising behavior that classical theory doesn’t predict; cryptanalysts must be cautious in assuming transferability. For design, this suggests ARX/Feistel round functions introduce an extra layer of security against straightforward neural transfer attacks.”
   - Possibly pose a question for future: e.g., “Can we design networks to avoid anti-transfer, or to explicitly correct it?”
   - The narrative culminates by resolving the tension: we expected (classically) positive transfer, but we found anti-transfer. Then we resolve that tension with theory and controls, leaving the reader convinced.

**Disconnected or Interrupting Sections:** 
- The paper already follows a logical order for the most part. One possible slight disconnect: the architecture-comparison table (Table 2) in Sec5.1 might distract before fully analyzing signal composition. It could possibly be moved to the end of Sec5.1 or Sec5.1 conclusion. But it’s not severely out of place.
- The final “Structural Controls” is quite long – a reader might start skimming. It might help to signpost its purpose (“We now rule out alternative explanations…”). A short intro sentence to 5.4 could help with flow.
- The jump from heavy empirical results to the formal theorem is fine, but maybe add one bridging sentence at start of 6.1: “Our empirical finding suggests a violation of classical Markov assumption. We now formalize when transfer should or should not occur.” This exists as “From Markov to Transfer: A Formal Theorem”, which is good.

**Narrative Redesign Summary:** The paper should build tension by first showing expected behavior (SPN positive transfer), then surprising behavior (ARX/Feistel anti-transfer), then explaining it. The evidence for the paradox (Sec5.3) should create the key “oh!” moment. The formal discussion then resolves the mystery, satisfying the reader’s need for an explanation.

## Part 6: Reviewer Psychology

As a cryptography PC member reading this draft, here are likely pain points and proposed fixes:

- **Point of Drop-off / “Stop Reading”:** Reviewers may start skimming if the introduction is not compelling. The overly detailed list of contributions with little high-level context could lose them early. *Fix:* Refine intro to immediately highlight the surprising paradox up front, then justify contributions.

- **Skepticism about Claims:**
  - **Anti-Transfer is real:** A reviewer might initially think “maybe this is just random error or sign-flip triviality.” They will look for your controls. The draft does a good job adding controls, which is positive. However, any hint of arbitrary parameter (like specific ∆P) might cause doubt. The mention of evolutionary search addresses that, but maybe a stronger statement like “observed consistently across ∆P and random seeds” could pre-empt skepticism.
  - **“Seven architectures” claim:** Reviewers will wonder which 7 and if they cover enough variety. The paper lists them in Methodology and shows one table; that’s good. But the phrase “≤2.6 pp spread across seven models” in contrib sounds definitive. Reviewers will likely verify from Table 2: but that table only shows 6 models (MLP, CNN, LSTM, GRU, Res. CNN, Siamese, plus the “ResNet” outside). Ensure all used architectures are clear. Possibly state which architectures (e.g., “we tested 6 additional models plus ResNet, see Table 2”).
  - **Statistical significance:** If any result is borderline, a reviewer might question significance. They gave t-test p-values (e.g. p<10^-7), which is very rigorous. But ensure the methodology for these tests is clear (they mention one-sample t-test). A potential worry: with 5 seeds, how do they do a t-test? They should clarify they did multiple runs or some bootstrap. If it's just 5 seeds, a normal t-test might be underpowered. Perhaps say “we perform bootstrapping with replacement 10,000 times and derive significance.” If not, at least say “over 5 seeds” might not justify a t-test directly – though they do mention one-sample t-test in PF. Some reviewers might say 5 values is too few for a t-test. If so, consider simply stating accuracy differences with confidence intervals (they do mention bootstrap CI).
  - **Citation demands:** A reviewer might want citations for any generalized claim. For example:
    - “the cipher is memoryless”: Not usually a reference needed, but they do prove it. If it was a known fact, a citation could be nice, but since they did the test, it's fine.
    - “Gohr’s ResNet” numbers: They cite [13] for Feistel DDT proof, but for the claim “Gohr’s numbers for Speck 8r, etc.” they cite [12]. That’s fine.
    - If they mention something like “no prior work...” a reviewer might think of counterexamples. They did cite [2] correctly.

- **“Belongs in Appendix”:** Reviewers often think some parts could be relegated to Appendix. 
  - The **algorithm pseudo-code** (Algorithm 1) is a small detail and might be unnecessary if the process can be described in text. They might say “we know how cross-round eval works – can move algorithm to appendix”.
  - The **MINE equation** might be considered technical background that could go to appendix or a footnote. If space is tight, mention MINE qualitatively and cite [5], skipping the full formula.
  - The **proofs of propositions** are already in Appendix A, which is good. The main text references them (Appendix A).
  - The **architectures table** could be in main or appendix. It's relevant but could be summarized (they do quantify spread in text anyway).
  - The **detailed limitations list** (three paragraphs) is somewhat long for a conference paper. It might be pushed to a “Discussion” appendix or trimmed. However, having some limitations in main is positive for reviewers, so perhaps keep but consider which parts are most important.
  - The **MINE calibration details** (“noise floor, upper reference”) might be more detail than needed. Could be briefer or appended.

- **Suspected Overclaiming:**
  - The phrase “foundational questions” could be seen as exaggeration if not justified. They should show these indeed affect many works (the survey [11] is used, so at least 66 papers surveyed).
  - Saying “anti-transfer is a *paradox*” might raise eyebrows. They do formalize it and show examples, so it’s not unfounded. Still, maybe they should emphasize it’s a surprising phenomenon rather than inherently mystical.
  - Theorem language “rigorously prove” is fine because they do have a proof in Appendix. Just ensure they say the assumptions clearly.

- **Points of Skepticism / Criticism:** 
  - “Is this specific to small block size (32-bit)?” The authors pre-empted with “block size 32, expect same for larger.” But a reviewer might press: Are we sure? Possibly mention if any smaller test on a toy 64-bit cipher was done (even if theoretical reasoning suggests yes).
  - “Is this just flipping of labels?” They actually test flipping (negated-output control) and show it’s not equivalent. That’s good. A reviewer might want those results explicitly (they give numbers).
  - “What about multi-pair distinguishers?” They note this in limitations, should clarify it more: for example, “We use single-pair MLPs; if one aggregates multiple pairs (like in [22]), does anti-transfer persist? We expect yes, since signal sign inversion doesn’t depend on number of pairs.” This is plausibly correct but a reviewer might still wonder. They could add a sentence to limitations about multi-pair networks being interesting future work.

- **Demand for Citations (again):** 
  - In discussion 6.2, they say “Gohr et al [13] proved Feistel distinguishers learn only DDT features.” [13] is likely that reference. Good.
  - The new Lemma 1 is proved in text; that’s fine. 
  - They might ask for a reference or footnote for “handy fact: output difference of a&b formula is (a&∆b)⊕(∆a&b)⊕(∆a&∆b)” (though they just proved it).
  - The proof of Markov property via VAE is compelling but new. It’s okay.

- **Reviewer Emotions:**
  - When seeing the puzzle (below chance performance with high MI), a reviewer might be intrigued and ready to believe the result, but will want assurance it's not a silly artifact. The controls are thus crucial and need to be clearly described. 
  - If the writing is unclear or the tables hard to parse, the reviewer may gloss over the results, hurting the paper’s perceived validity. For example, Table 1 has some rows with blanks ("–") which they should ensure is explained (I see blank entries, presumably meaning “not applicable?”). Clarify in caption if needed.
  - The use of multiple decimals (like “52.47 ± 0.18 [52.30,52.71]”) in table for low-distinguish accuracy could appear overly precise. Usually we round to 2 decimals. It’s not a big deal, but consistency helps.

**Proposed fixes to address these concerns:**

- Emphasize the novelty without hype. Possibly start Discussion with a crisp statement: “The fact that Feistel networks **actively invert** the signal is a new discovery.” This primes reviewers to appreciate it.
- Simplify any part where a reviewer might get lost. For instance, maybe add a small sentence in Sec5.2: “Thus we measure *transfer polarity* as accuracy above 50% vs below 50%.” The term “transfer polarity” is used but not defined.
- In methodology or results, clarify any potential confusion: e.g., for cross-cipher transfer, specify “Speck→Simon: model trained on Speck tested on Simon data.” 
- Check all figures/tables are self-contained and clear (captions must explain what is shown without needing back-reference).
- For any strong numeric claim (like significance), ensure the methodology is bulletproof or soften the wording if needed (“we estimate significance by [method], results strongly indicate...”).
- Make sure to state any assumptions clearly (e.g., independent keys vs IRK) and highlight that we tested IRK vs standard key-schedule. They did that in controls.

## Part 7: Acceptance-Oriented Rewrite Plan

We assume the **scientific results are fixed**. We can only change writing, organization, framing, etc. We must respect page limits (20 + 10). Suggested changes are ranked by *acceptance impact per page* used.

### High-Impact Changes

1. **Sharpen and Shorten the Introduction (Cost: ~0.5–1.0 page)** – *High Impact:* The intro must quickly convey motivation, questions, and main results. Rewriting it to emphasize the surprising anti-transfer phenomenon (as noted in Part 6) will hook reviewers. Remove non-essential narrative (e.g. “billion devices”) and break into clearer paragraphs. Clearly state the two research questions and provide an “organizational roadmap” sentence at the end. **Space cost:** Might actually reduce the intro by ~1/2 page by cutting fluff and bullet details. This will pay off by preventing reviewer fatigue early. *Current content to cut/shorten:* The broad IoT motivation, lengthy contributions list details, redundant framing.

2. **Add Explicit Organizational Outline (Cost: 0.25 page)** – *High Impact:* Including a sentence like “Sec.2 reviews related work; Sec.3 formalizes the model; Sec.4 methodology; Sec.5 presents experiments; Sec.6 discussion; Sec.7 concludes” helps reviewers see the paper’s structure. It’s a small addition but aids navigation. The intro or start of each section should then flow.

3. **Consolidate Sections/Remove Redundancy (Cost: 0 pages, just restructuring)** – *High Impact:* Merge related subsections where possible to avoid fragmentation. For instance:
   - Merge 4.1 and 4.2 into “Data and Input Representation”.
   - Combine 4.5/4.6/4.7 under “Analysis Tools” with sub-headers (or fold their content into one unnumbered paragraph each).
   - Combine findings narrative in Section 5 with fewer breaks (the subsections are fine, but ensure each flows as part of a continuous story with clear transitions). 

   This saves space by reducing heading overhead and redundant explanation. For example, some lines of prose around table discussions can be tightened. **Space cost:** We might free up ~0.5 page from combining subsections and reducing repeated phrases like “We evaluate…” at the start of paragraphs.

4. **Move Detailed Content to Appendix (Cost: reduce main by ~1–2 pages)** – *High Impact:* Identify parts that, if moved, streamline the main text without losing essential info:
   - **Algorithm 1:** This procedural detail can be described in words or Appendix. Unless the algorithm is crucial to the storyline (it’s basically the experiment loop), it can be removed or replaced with a sentence. *Benefit:* Saves roughly 0.5 page.
   - **MINE Formula & Derivation:** The explicit equation (2) and extended explanation might be shortened or moved to appendix. Just say “We compute a neural MI lower bound [5] (see App.)” and move the math out. *Benefit:* ~0.5 page.
   - **Long Experimental Controls Text:** As noted, much of Section 5.4 could be condensed by summarizing results in a table or bullet list, with the detailed justification in an Appendix paragraph. Or split into “main text conclusion, Appendix details.” *Benefit:* Possibly ~0.5–1.0 page.

   Moving such content allows adding more narrative or rephrasing in the main space that reviewers see.

5. **Refine Figures and Tables (Cost: ~0 pages, reformatting)** – *High Impact:* Ensure all figures/tables have clear captions and are referred to explicitly in text. Combine tables if possible (e.g., Table 1 and the architecture table are both in Sec5.1; maybe make one composite table or move architecture results to an appendix table, summarizing only key spread in text). If any figure/table is redundant or low-impact, consider removing or moving it (given space constraints). Well-presented tables make results more convincing. *Space:* Possibly save ~0–0.5 page by eliminating redundant columns or compressing.

6. **Rewrite Long Sentences / Paragraphs (Cost: ~0 pages, rewrite)** – *High Impact:* Although this doesn't change length much, it improves clarity significantly, thus making reviewers’ job easier. For example, break intro bullet items into multiple sentences, clarify methods as indicated in Part 4. Each section should have a clear first sentence introducing its content. This may take no extra pages, but time investment in rewriting. 

### Medium-Impact Changes

1. **Clarify/Define Jargon on First Use (Cost: ~0.2 page)** – *Medium Impact:* Add brief definitions (possibly parentheses) for terms like DDT, saliency, “source-round model,” etc. This ensures no reviewer pauses to guess meaning. Place definitions seamlessly (e.g. “the difference distribution table (DDT)”).

2. **Statistical Methods Paragraph (Cost: 0.1–0.2 page)** – *Medium Impact:* While not huge space, explicitly stating “We perform n=5 trials per configuration, reporting mean±std. We apply bootstrapping to compute 95% CIs for boundary cases. Significance is tested by one-sample t-test vs 0.5, Bonferroni-adjusted.” This addresses any doubt about the validity of claims. It can fit in Methodology or in a short subparagraph.

3. **Add “Future Work” Pointers (Cost: 0.1 page)** – *Medium Impact:* A couple of sentences (or bullet points) in Conclusion or discussion listing concrete next steps (e.g., test larger ciphers, multi-pair nets, practical key-recovery experiments). This shows reviewers the authors are thinking ahead and acknowledges open problems. It was partially present but could be explicit.

4. **Tighten Related Work (Cost: -0.5 page)** – *Medium Impact:* Remove or drastically shorten parts that are not crucial. For example, the “Cipher-specific improvements” can be replaced by one sentence. The list of seven architectures in related work perhaps also can be trimmed or moved. This saves space to reallocate to narrative or justification. 

5. **Improve Figures (Cost: ~0 pages, possibly adjustment)** – *Medium Impact:* Make sure Fig.1 is referenced in text and explained early (“Figure 1 illustrates the positive vs anti-transfer scenario”). If fig1 is not referenced, add a mention. Add legends if missing (the extraction didn’t show, but ensure axis labels etc. are clear). Reviewers often look at figures first, so they must stand alone well. If any figure is unclear, refine or add a mini-description in text.

### Low-Impact Cosmetic Changes

1. **Terminology Consistency:** Ensure terms are used consistently (e.g., use “anti-transfer” vs “anti‑transfer” uniformly). Check hyphenation (“cross-round” vs “cross round”). Standardize notation (e.g., sometimes “r′” is written, sometimes “r'”). Minimal space cost.

2. **Remove Redundant Words:** Small cuts like “compute SmoothGrad [19] saliency maps over the input bit vector” could be “compute SmoothGrad saliency over the input bits.” Tiny space save.

3. **Edit for Grammar / Typos:** Correct any OCR artifacts (“dierential” should be “differential” everywhere, “” to normal quotes or dashes). This mostly doesn’t affect page count but is needed polish.

4. **Citation Formatting:** Ensure all [refs] are properly formatted and updated. (This is standard practice, but minor for acceptance.)

5. **Color and Layout:** If any color in figures is not printer-friendly (assuming CANS might be printed B/W), adjust. Not page count but helps readability.

### Page Budget Allocation (Main Paper 20 pages)

Here’s a proposed split (subject to adjustment):

- **Title/Abstract/Intro:** ~1.5 pages (Abstract ≈0.5, Intro ≈1.0 after pruning).
- **Related Work:** ~1.5 pages.
- **Prelim/Problem Formulation:** ~0.75 pages.
- **Methodology:** ~2.5–3.0 pages (with some moved to appendix).
- **Experiments (Sec5):** ~7.0 pages (this is heavy; may need to trim a bit, possibly by moving some control details to appendix).
- **Discussion (Sec6):** ~4.0 pages.
- **Conclusion:** ~0.75 page.
- **Total:** ~18–19 pages for main content. That leaves a ~1–2 page buffer for diagrams, or small expansion if needed. If we can condense another 0.5 page from results/discussion, it’s safe.

**Keep / Cut / Move Table (Main vs. Appendix):**

| Content                          | Keep in Main (✔) | Move to Appendix (↗) | Cut (✗)       | Comments/ Rationale                                 |
|----------------------------------|:----------------:|:--------------------:|:-------------:|-----------------------------------------------------|
| Abstract (concise)               | ✔                |                      |               | Essential                                            |
| Intro Motivation (trimmed)       | ✔                |                      |               | Include only necessary context                      |
| Contributions list               | ✔ (shortened)    |                      |               | Important, but shorten/clarify bullets              |
| Related Work (tightened)         | ✔                |                      |               | Remove cipher-specific details (concise summary)    |
| Threat Model / Problem (short)   | ✔                |                      |               | Keep                                                 |
| Data generation details (4.1)    | ✔ (summarize)    | ↗ (full hyperparams) |               | Key points in main, all hyperparams in Appendix     |
| Input rep (4.2)                  | ✔                |                      |               | Keep R2_diff explanation (important)               |
| Architecture list (4.3)          | ✔ (mention CNN,RNN etc) | ↗ (specific configs) |     | Main: mention architectures tested; details in App   |
| Classical baseline (4.4)         | ✔ (brief)        |                      |               | Keep brief statement of method                     |
| MINE / Saliency / ∆P (4.5-4.7)   | ✔ (shorter)     | ↗ (detailed eq/params) |             | Move formula and some nitty-gritty to Appendix      |
| Exp. Table 1 (accuracy vs rounds)| ✔ (maybe split)  |                      |               | Keep (key result)                                  |
| Exp. Table 2 (architectures)     | ✔ or ↗ (if space)|                      |               | Can move to App if needed (just mention spread)    |
| Exp. Table 3 (transfer accuracy) | ✔                |                      |               | Keep (addresses main question)                     |
| Figures 3-5 (MI vs acc, saliency, VAE) | ✔         |                      |               | Keep (illustrative evidence)                        |
| Definition 1 (paradox)           | ✔                |                      |               | Keep (formal core concept)                          |
| Sec 5.4 sub-experiments (i-vi)   | ✔ (summary)      | ↗ (raw data, extra plots) |            | Main: summary statements; full tables/numbers to App|
| Algorithm 1                      | ↗                |                      |               | Move or remove (describe in text instead)          |
| Discussion Theorem 1 (statement) | ✔                |                      |               | Keep (key theory)                                  |
| Discussion proofs (Prop3, etc)   | ↗                |                      |               | Proofs in Appendix                                 |
| Discussion 6.2 Lemma and bullets | ✔ (short)       | ↗ (detailed bias data) |            | Summary in text, or move bias tables to App         |
| Limitations (6.3)                | ✔                |                      |               | Important to keep some, but could trim examples    |
| Conclusion                       | ✔                |                      |               | Keep                                              |
| References                       | (excluded from 20pp) |                |               | Provide minimal needed, maybe move extended citations list to Appendix if page count strict|

**Ranked List of Changes by Acceptance Gain per Page Cost:**

1. **Rework the Introduction** – *Gain: High.* Present surprising findings early, clarify questions. (Cost ~-0.5 to 1 page).
2. **Add Organization Roadmap** – *Gain: High.* (Cost ~0.25 page)
3. **Trim Related Work / Prelims** – *Gain: Medium-High.* Remove tangential details. (Gain pages ~0.5)
4. **Shorten Methodology sub-sections (esp. formulaic parts)** – *Gain: Medium.* (Gain ~0.5 page, by moving math to appendix).
5. **Consolidate Experiment Results (remove fluff)** – *Gain: Medium.* Emphasize main findings, not every detail. (Gain ~0.5 page).
6. **Improve Narrative Flow (adding lead-ins)** – *Gain: Medium.* Clear transitions (no extra pages, just rewriting).
7. **Clarify Statistical Approach** – *Gain: Medium.* (Cost ~0.1 page).
8. **Move Algorithm to Appendix** – *Gain: Low-Medium.* (Gain ~0.5 page, since algorithm box likely ~0.3–0.5p).
9. **Rephrase Hype / Informal Phrasings** – *Gain: Medium-Low.* (Rewrite, no page cost).
10. **Restructure Discussion/Limitations (if needed)** – *Gain: Low-Medium.* (May shorten, but already fine).
11. **Add Future Work lines** – *Gain: Low.* (few lines).
12. **Consistent Formatting / Typos** – *Gain: Low.* (Cosmetic, no space cost).
13. **Mention Code Release** – *Gain: Low.* (One line in footnote or conclusion).
14. **Figure/Table Formatting** – *Gain: Low.* (No space change).
15. **Appendix Cleanup** – *Gain: Low.* (No change in main).

Before changes, estimate acceptance probability: Likely around **maybe 40-50%** (a “borderline” accept-reject) if it were at CANS now, given strong results but rough presentation. After revisions as above, it could rise to **70-80%** (Strong accept) because clarity will let reviewers fully appreciate the novelty and rigor.

## Part 8: Brutal Verdict

**Score if Unchanged (CANS style):** Probably around **3 (Weak Accept/Weak Reject)** on a typical 1–6 scale, or “Marginal” / Score ~4/6. The core results are interesting, but the writing and structure would irritate reviewers. Many would say “Good results, but the paper is messy and too long.” Some might reject on the basis of unclear presentation or missing polish. If CANS has a 1-5 scale (with 3=borderline), this might score 3.

**Key Structural/Writing Changes for Acceptance:**
- **Sharper Opening:** Lead with the puzzle/ surprise to immediately convince reviewers this is worth reading. Don’t bury it in dense prose.
- **Tight Organization:** Add outlines and smooth transitions so reviewers don’t get lost. They must quickly see the flow and logic of the paper.
- **Conciseness:** Cut irrelevant or peripheral content (like deep related-work digressions, over-detailed method specs) and move it to appendix. Keep the main narrative lean.
- **Clarity of Claims:** Tone down any grandiose claims or language, and back up strong claims with citations or clearly labeled evidence. Present numbers with appropriate uncertainty and describe significance carefully.
- **Focus on Story:** Emphasize *why* each experiment is done and *what it means*. This narrative emphasis will keep reviewers engaged instead of skimming.
- **Fix Errors/Formatting:** Typos, weird symbols (like `pp1`), and heading anomalies must be corrected. Presentation matters to busy reviewers.
- **Add Missing Pieces:** A brief prelim with definitions, a clear statement of organization, and concise statement of statistical methods will avoid petty criticisms.
- **Highlight Implications:** Explicitly state the impact on neural cryptanalysis practice and cipher design, so reviewers see the broader significance.

If the paper remains technically the same but the above fixes are applied within the 20-page limit, reviewers will focus on the novel scientific findings instead of being distracted by poor exposition. The acceptance chances will then reflect the true quality of the work.