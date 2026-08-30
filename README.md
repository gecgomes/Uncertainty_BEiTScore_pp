# BEiTScore++

### Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

[Gonçalo Gomes](mailto:goncaloecgomes@tecnico.ulisboa.pt)<sup>1,2</sup>, [Bruno Martins](mailto:bruno.g.martins@tecnico.ulisboa.pt)<sup>1,2</sup>, [Chrysoula Zerva](mailto:chrysoula.zerva@tecnico.ulisboa.pt)<sup>1,3</sup>

<sup>1</sup> Instituto Superior Técnico, University of Lisbon &nbsp;&nbsp;·&nbsp;&nbsp; <sup>2</sup> INESC-ID &nbsp;&nbsp;·&nbsp;&nbsp; <sup>3</sup> Instituto de Telecomunicações

---

## Code coming soon

This repository will host the source code, trained model checkpoints, and evaluation scripts for BEiTScore++.

The release is being prepared and will be published here shortly. Star or watch the repository to be notified when it lands.

---

## Abstract

Reliable uncertainty estimation is important for deploying evaluation metrics in real-world settings, where both input ambiguity and domain shifts are common. In this work, we propose an uncertainty-aware extension of the BEiTScore image captioning metric, producing calibrated confidence intervals alongside alignment scores. Our approach models predictive distributions over human judgments, enabling both accurate scoring and principled uncertainty quantification. We evaluate our BEiTScore++ metric across multiple benchmarks with human-annotated ratings, demonstrating strong performance in terms of ranking correlation, calibration, and interval sharpness. BEiTScore++ not only produces accurate predictions, but also captures aleatoric uncertainty arising from ambiguous inputs, and epistemic uncertainty due to limited model knowledge, making it a reliable tool for uncertainty-aware evaluation of image-caption alignment.

## Overview

BEiTScore++ formulates reference-free image captioning evaluation as a quantile regression problem. A BEiT-3 cross-encoder backbone feeds a quantile regression head with 11 output nodes, predicting a central score at the 50th percentile together with five nested centered prediction intervals (70%, 75%, 80%, 85%, and 90%).

The main components are the following:

- Monotone quantile construction. The median is predicted directly, and the remaining quantiles are built iteratively from the center using bounded multiplicative offsets. This enforces quantile monotonicity by design, preventing quantile crossing and keeping all outputs within the [0, 1] score range.
- Centered interval loss. Instead of the standard pinball loss, training uses the centered interval objective of Chung et al. (2021), which explicitly balances coverage calibration and interval sharpness, combined with an L1 loss on the median.
- Conformal calibration. A post-hoc conformal risk control step calibrates the intervals through two parameters, a multiplicative factor and an additive offset, applied around the median.
- Subgroup-based calibration. Training and calibration data are partitioned into homogeneous subgroups via k-means over image embeddings. Mini-batches are drawn from a single subgroup at a time, and conformal calibration is performed per subgroup, retaining the most conservative parameters to better approximate conditional coverage.

## Planned contents of this repository

- Model definition for the BEiT-3 backbone and the chained quantile regression head, in base and large variants
- Training code, including the centered interval loss and the subgroup batching strategy
- Post-hoc conformal calibration scripts
- Inference and scoring utilities for producing scores together with confidence intervals
- Evaluation scripts reproducing the reported coverage, Kendall's tau-c, sharpness, and Uncertainty Pearson Score (UPS) results
- Scripts for the uncertainty analyses under image degradations and under out-of-domain artwork data
- Released checkpoints

## Data

Training uses a mixture of datasets with human-annotated image-caption alignment scores, namely VICR, Polaris, Nebula, and EvalMuse-40k, with the human judgments normalized to the [0, 1] range.

Evaluation covers VICR, Polaris, Nebula, Flickr8K-Expert, Flickr8K-CF, and Composite. The analysis of the different sources of uncertainty additionally uses in-domain data from IIW, DOCCI, Localized Narratives, and MSCOCO, together with the out-of-domain artwork datasets Docent and ArtCaps.

## Key findings

- Correlation with human judgments is preserved at the level of the original BEiTScore metric, so adding uncertainty estimation does not degrade scoring quality.
- The chained quantile head paired with the centered interval loss improves UPS and sharpness across all benchmarks, relative to independent quantile heads trained with the pinball loss.
- Subgroup-based calibration yields substantial coverage gains after conformal calibration, particularly on out-of-distribution benchmarks.
- Compared to a Monte Carlo dropout baseline, the method gives better uncertainty quality while requiring a single forward pass at inference.
- Predicted interval widths grow with the severity of image degradations such as Gaussian blur, and are consistently larger on out-of-domain artwork images, reflecting aleatoric and epistemic uncertainty respectively.

## Citation

If you find this work useful, please consider citing it. The full reference will be updated once the paper is publicly available.

```bibtex
@misc{gomes2026beitscorepp,
  title  = {BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression},
  author = {Gomes, Gonçalo and Martins, Bruno and Zerva, Chrysoula},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgements

This work was supported by Fundação para a Ciência e a Tecnologia through the doctoral grant.

## Contact

For questions about the paper or the upcoming code release, please open an issue or contact goncaloecgomes@tecnico.ulisboa.pt.

## License

To be defined upon release.
