# ICLR 2027 manuscript build

This source bundle uses the official ICLR 2027 LaTeX style without modifying
its formatting parameters.

## Build

Run the following commands from the `paper/` directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error main_iclr2027.tex
bibtex main_iclr2027
pdflatex -interaction=nonstopmode -halt-on-error main_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error main_iclr2027.tex
```

The submission remains anonymous because `\iclrfinalcopy` is commented out.
The main scientific text ends on page 9, followed by the mandatory
AI-use, ethics, and reproducibility statements and references. Appendices begin
after the references. The main text remains within the 9-page limit.

The manuscript is organized as an intervention audit: it separates detection,
weak-group allocation, post-retraining utility, and transfer; treats RV-Q as an
initial value-alignment baseline; and gives equal visibility to positive and
negative evidence.

## Included dependencies

- `main_iclr2027.tex`
- `references.bib`
- the official `iclr2027_conference.sty` and bibliography style files
- the intervention-audit overview figure
- the exploratory modern-attribution table included by the appendix
- the baseline-supplement tables (absolute references, 2% budget detail,
  all-budget three-dataset matrix, model-baseline/JTT diagnostics)
- the exploratory RV-Q validation-sensitivity audit (tail fraction,
  validation-pool size, and validation-label quality) with its appendix table
- the 20-seed Waterbirds full-retraining comparison at the best observed
  `tau=0.35`; the complete full-retraining tail curve, frozen-feature extension,
  and additional end-to-end blocks are in the appendices
- the Waterbirds RV-Q-Gated seeds 40--59 tail-fraction ablation in the
  appendix; its displayed tau is selected by mean validation balanced accuracy
  without using test WGA, and the complete sensitivity curve and table are
  generated from the local frozen-feature output
- the independent CelebA Young/Male full-retraining comparison at the fixed
  development-selected `tau=0.50`, with the complete tail-fraction sensitivity
  curve in the appendix

## Baseline supplement

The baseline supplement extends the frozen audit with no-query model references
(ERM, oracle GroupDRO, JTT diagnostic) and mainstream query rules (in-sample
Cleanlab, five-fold OOF Cleanlab, ActiveClean-style expected model change,
Active Label Cleaning). All rows share the same feature and retraining
evaluation, and the complete tables remain in Appendix C.12.

## Alignment and failure sensitivity

The appendix reports a one-factor, ten-seed CelebA audit after frozen-feature
linear-head retraining. It varies the tail fraction and the size and quality of
the validation pool to test when the value surrogate ceases to align with the
repair endpoint.

## 20-seed Waterbirds transfer audits

The frozen-feature extension uses seeds 0--19. The main manuscript reports the
best full-retraining tail setting, `tau=0.35`, on seeds 20--39, where Random,
Loss, TracIn, NoiseScore, and RV-Q share the same training setup. The appendix
reports the complete full-retraining tail sensitivity, the frozen-feature
sensitivity table, and the additional Waterbirds end-to-end blocks.
