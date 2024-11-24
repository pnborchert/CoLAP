# Embedding Analysis

In the following, we compare the embedding spaces of the CoLAP and PCT models trained on the English XNLI dataset and fine-tuned for few-shot learning in target languages. These models are the same ones used to report cross-lingual transfer performance in Table 1.

To compare the contextualized representations, we randomly sampled instances from the XNLI test set in the respective target languages (German and Swahili) as well as English. We extracted the last hidden representation of the prompting token (`<mask>` for XLM-R, `<eos>` for Mistral and Gemma).

These hidden representations were projected into a two-dimensional space using t-SNE. We evaluated the representation spaces in both a lower-resource setting (K=5) and a higher-resource setting (K=250). The analyses were conducted for a high-resource language (German) and a low-resource language (Swahili).

**Findings:**
The t-SNE visualizations reveal that CoLAP consistently produces more tightly clustered multilingual representations for the same class, while maintaining well-separated clusters for different classes across both few-shot settings and different languages. These findings demonstrate CoLAP's ability to generate a more task-discriminative representation space, which aligns with its performance improvements reported in the paper.


| **Figure 1: K=5, languages={German, English}** | |
| -------- | ------- |
| CoLAP XCCL    | PCT |
| ![](figures/tsne-xnli-xlm-roberta-base-de-colap-5-xccl.png)  | ![](figures/tsne-xnli-xlm-roberta-base-de-pct-5.png)    |


| **Figure 2: K=5, languages={Swahili, English}** | |
| -------- | ------- |
| CoLAP XCCL    | PCT |
| ![](figures/tsne-xnli-xlm-roberta-base-sw-colap-5-xccl.png)  | ![](figures/tsne-xnli-xlm-roberta-base-sw-pct-5.png)    |


| **Figure 3: K=250, languages={German, English}** | |
| -------- | ------- |
| CoLAP XCCL    | PCT |
| ![](figures/tsne-xnli-xlm-roberta-base-de-colap-250-xccl.png)  | ![](figures/tsne-xnli-xlm-roberta-base-de-pct-250.png)    |

| **Figure 4: K=250, languages={Swahili, English}** | |
| -------- | ------- |
| CoLAP XCCL    | PCT |
| ![](figures/tsne-xnli-xlm-roberta-base-sw-colap-250-xccl.png)  | ![](figures/tsne-xnli-xlm-roberta-base-sw-pct-250.png)    |
