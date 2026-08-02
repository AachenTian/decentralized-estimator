# Phase 5: Formal Training, Evaluation, Checkpointing, Resume

## Default formal schedule

Per owner:

- 200 synchronization rounds;
- 12 real environments;
- 25 transitions per environment;
- 300 private real transitions per round;
- 50 five-member dynamics updates per round;
- 1024 model-rollout starting states;
- model horizon scheduled from 1 to 6;
- 100 critic/actor/alpha update cycles per round;
- SAC minibatch: 50% private real and 50% private model data.

System-wide formal real training cost:

```text
200 x 3 owners x 12 envs x 25 steps = 180,000
```

The separate common actor-normalization calibration uses:

```text
4 x 12 x 25 = 1,200
```

Evaluation interaction is logged separately and is not included in the training
environment-step counter.

## Checkpoints

`checkpoints/latest.pkl` contains all three independent learners, all three
dynamics ensembles, private replay contents, model replay contents, environment
carry states, counters, and the common actor normalizer.

`checkpoints/best.pkl` is lightweight and contains the best three-actor tuple
plus normalization and evaluation metadata.

Numbered `actors_round_XXXX.pkl` files are lightweight actor checkpoints for
training-process animation.

## Commands

```bash
bash scripts/test.sh
SEED=0 bash scripts/train_formal.sh
```

Resume:

```bash
SEED=0 bash scripts/resume_formal.sh
```
