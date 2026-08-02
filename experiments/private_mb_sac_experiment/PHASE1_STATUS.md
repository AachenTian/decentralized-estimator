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

## Terminal compatibility fix

The collector now treats `next_state.step >= max_steps` as a terminal
transition in addition to the native JaxMARL `episode_done` flag. This handles
JaxMARL releases that expose the state at step 25 but only report `__all__`
after a subsequent call.

For the standard smoke test (`2 envs x 25 steps`), each owner must now report
exactly two terminal transitions. The executable raises an error if this
invariant is violated.
