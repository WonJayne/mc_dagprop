# OpenBus fixed-precedence integration contract

`mc_dagprop` evaluates a fixed event-activity DAG. OpenBus constructs that DAG upstream, detects infrastructure/resource conflicts, and exports the selected conflict orders as deterministic or stochastic precedence/separation activities.

`PropagationContext` must not contain unresolved disjunctions. To evaluate a different conflict order, OpenBus must build a different graph.

A resource precedence activity is just an activity:

- `source`: release event of the preceding train/resource occupation;
- `target`: acquire event of the following train/resource occupation;
- `minimal_duration`: separation or reoccupation time;
- `activity_type`: may be unregistered, making the stochastic extra delay deterministic zero.

Unregistered activity types therefore still contribute deterministic `minimal_duration`; registered delay families add stochastic extra delay only. The graph must be acyclic.

The current kernel evaluates fixed precedence graphs. Dynamic conflict-order selection and dispatching policies are outside `mc_dagprop` and must be represented by alternative graphs or future policy layers.
