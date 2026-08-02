# Phase 1 status

Implemented:

- configuration validation;
- actor initialization and snapshot exchange;
- common environment RNG and owner-private policy RNG;
- three isolated real replay buffers;
- vectorized real-environment collection;
- terminal-before-reset storage;
- nonterminal dynamics sampling filter;
- interaction counters;
- offline W&B collection metrics;
- executable smoke test.

Not implemented yet:

- calibrated frozen actor observation normalization;
- five-member dynamics training;
- private model rollout;
- twin-critic SAC updates;
- actor/alpha updates;
- evaluation and checkpointing.
