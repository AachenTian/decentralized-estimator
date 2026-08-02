# Phase 4: Three Independent Private SAC Learners

## Implemented

Each owner now has a completely independent:

- live Gaussian actor and Adam optimizer state;
- centralized twin online critic and optimizer state;
- twin target critic parameters;
- entropy temperature and optimizer state;
- fixed diagnostic real batch;
- critic and actor update counters.

The three owners continue to have independent real replay, model replay,
dynamics ensemble, dynamics normalizer, and random streams.

## One synchronization round

1. Copy all three current live actors into one immutable snapshot bank.
2. Use this same bank for all three private real collectors.
3. Train each owner's five-member dynamics ensemble from its private replay.
4. Generate each owner's private model rollout with the same frozen bank.
5. Update each owner locally:
   - focal action comes from its live actor and carries gradient;
   - opponent actions come from the frozen bank and use stop-gradient;
   - only the focal live actor is updated.
6. Exchange newly updated actors only at the next round.

There is no actor averaging and no replay sharing.

## Phase-four smoke settings

Per owner:

- 50 real transitions;
- 10 dynamics updates;
- up to 64 synthetic transitions;
- 10 critic updates;
- 10 actor/alpha updates with policy delay 1.

The smoke test validates finite losses, update counts, actor movement away from
the round snapshot, snapshot immutability, and private runtime/replay isolation.

## Not yet implemented

The phase-four package is a core integration smoke test. Formal 200-round
training still needs:

- deterministic multi-seed evaluation;
- best/latest checkpoint persistence and resume;
- scheduled horizon 1-to-6 in the formal loop;
- formal 50 dynamics and 100 SAC updates per round;
- final W&B artifact upload;
- animation of the trained three-actor tuple.
