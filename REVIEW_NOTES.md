# Review notes for the notebook owner

These points were identified by reading the supplied notebook; no GPU retraining or independent reproduction was performed during the repository preparation.

1. **Code structure works in the uploaded run.** A two-convolution CNN, label mapping, train/validation split, white-box FGSM, a 10-step PGD validation experiment, 40-step PGD with 3 restarts, and held-out test evaluation are present and have recorded outputs.
2. **The results are preserved as reported** in the notebook. Held-out results: clean 90.72/91.93, FGSM 52.51/72.33, PGD 32.30/62.50 (baseline/defended). No figures were fabricated; the three PNG figures were extracted from notebook outputs.
3. **Important causal limitation:** The defended model received an extra manual batch and full fine-tuning epoch, while the baseline did not. The difference cannot be attributed solely to adversarial training. Ideally add an equal-budget clean-only fine-tuning control *before interpreting the gain causally*.
4. **Not exactly reproducible:** only the split and final test attack have explicit seeds. A fresh run may not give identical percentages. The original notebook was built incrementally; individual training cells can have effects if rerun.
5. **No checkpoints supplied:** both `.pth` files were mentioned and downloaded in Colab but were not included in the user's uploaded notebook file. Do not claim the packaged repository contains them.
6. **Technical precision:** FGSM and PGD use the model's input gradients, not weight gradients; PGD projection enforces an L-infinity budget, and the attacks optimize cross-entropy without guaranteed misclassification.
7. **Claims:** No novel method, certification, or generalized robustness claim should be made. The test-set 30.20 pp PGD improvement is an observational comparison under a specified attack with unequal training budgets.
8. **Potential future cleanup:** move Colab-specific file download cells to the notebook end; pin dependencies; make an end-to-end seeded runner; export confusion matrices and matched-control results.