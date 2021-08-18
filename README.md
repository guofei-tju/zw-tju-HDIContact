# HDIContact：a method for the prediction of residue–residue contacts on hetero-dimer interfaces from sequential information
Proteins maintain the functional order of cell in life by interacting with other proteins. The determination of protein complex structural information will gives biological insights for the research of diseases and drugs. Recently, a breakthrough has been made in protein monomer structure prediction. However, due to the limited number of the known protein structure and homologous sequences of complexes, the prediction of residue–residue contacts on hetero-dimer interfaces is still a challenge. In this study, we have developed a deep learning framework HDIContact for inferring inter-protein residue contacts from sequential information. We used pre-train protein language model to produce Multiple Sequence Alignment (MSA) two-dimensional (2D) embeddings based on patterns of concatenated MSA, which could reduce the influence of noise on MSA caused by mismatched sequences or less homology. For MSA 2D embeddings, HDIContact took advantage of Bi-directional Long Short-Term Memory (BiLSTM) to capture 2D context of residue pair from two different directions of the receptor and the ligand. Our comprehensive assessment on the Escherichia coli (E. coli) test dataset showed that HDIContact outperformed other state-of-the-art methods, with top precisions of 65.96%, the Area Under the Receiver Operating Characteristic curve (AUROC) of 83.08%, and the Area Under the Precision Recall curve (AUPR) of 25.02%. AUROC and AUPR were about 16.98% and 18.91% higher than other methods, respectively. Meanwhile, compared with other model architectures, AUROC and AUPR could increase at least 1.17% and 4.16%. The analysis about the influence of MSA and distance threshold on the methods further proved the more outstanding performance of our method. In addition, we also proved the potential of HDIContact for human-virus protein-protein complexes, by achieving top 5 precisions of 80% on protein complex O75475-P04584 related to Human Immunodeficiency Virus (HIV). All experiments indicate that our method is a valuable technical tool for predicting inter-protein residue contacts, which will be helpful for understanding protein-protein interaction mechanisms.
![](https://github.com/vivian2229/zw-tju-HDIContact/blob/main/Figure1-Framework.jpg)
