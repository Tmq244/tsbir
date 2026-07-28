# Report and Technical Notes

This directory collects the assessment report and the supporting technical documentation for the TASK-former reproduction.

## Submitted material

- [tsbir.pdf](tsbir.pdf): final experimental report.

## Technical documentation

- [optimization_experiments.md](optimization_experiments.md): complete baseline and four-method optimization record, including Human/Synthetic metrics and statistical analysis.
- [implementation_notes.md](implementation_notes.md): end-to-end reproduction notes covering data, training, and evaluation.
- [loss_and_gradient_notes.md](loss_and_gradient_notes.md): derivation of the three losses and their gradient paths.
- [gated_fusion_notes.md](gated_fusion_notes.md): design and implementation notes for learnable gated fusion.

## Training records

The [`training_runs`](training_runs/) directory contains the five formal runs: baseline average fusion, jointly trained Gate, frozen reliability Gate, segment dropout, and C2 human-style consistency. Each run provides its configuration, original training log, logged loss data, epoch summary, and loss-curve image. Smoke tests and aborted runs are excluded.

The technical notes are written in Chinese because they were produced as part of the experiment audit. The root [README](../README.md) is the English project entry point.
