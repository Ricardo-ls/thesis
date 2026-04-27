# Geometry Profiles Summary

This table compares the three Stage 3 geometry feasibility extensions on the clean-target feasible subset.

| geometry_profile | total_windows | feasible_windows | discarded_windows | retention_rate | main_constraint_type | source_family | window_violation_rate | infeasible_transition_rate | mean_infeasible_transitions_per_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| obstacle_v1 | 36073 | 25430 | 10643 | 0.704959 | central rectangular obstacle [1.2, 1.8] x [1.2, 1.8] | controlled_benchmark | 0.280771 | 0.075752 | 1.439284 |
| obstacle_v1 | 36073 | 25430 | 10643 | 0.704959 | central rectangular obstacle [1.2, 1.8] x [1.2, 1.8] | phase1_baseline | 0.002366 | 0.000321 | 0.006107 |
| obstacle_v1 | 36073 | 25430 | 10643 | 0.704959 | central rectangular obstacle [1.2, 1.8] x [1.2, 1.8] | refinement | 0.043657 | 0.012223 | 0.232237 |
| obstacle_v1 | 36073 | 25430 | 10643 | 0.704959 | central rectangular obstacle [1.2, 1.8] x [1.2, 1.8] | refinement_alpha_sweep | 0.043702 | 0.011027 | 0.209506 |
| two_room_v1 | 36073 | 27544 | 8529 | 0.763563 | internal wall with narrow opening y in [1.35, 1.65] | controlled_benchmark | 0.028096 | 0.002695 | 0.051209 |
| two_room_v1 | 36073 | 27544 | 8529 | 0.763563 | internal wall with narrow opening y in [1.35, 1.65] | phase1_baseline | 0.003925 | 0.000290 | 0.005507 |
| two_room_v1 | 36073 | 27544 | 8529 | 0.763563 | internal wall with narrow opening y in [1.35, 1.65] | refinement | 0.027671 | 0.002474 | 0.047004 |
| two_room_v1 | 36073 | 27544 | 8529 | 0.763563 | internal wall with narrow opening y in [1.35, 1.65] | refinement_alpha_sweep | 0.028727 | 0.002643 | 0.050216 |
| wall_door_v1 | 36073 | 30014 | 6059 | 0.832035 | internal wall with door opening y in [1.2, 1.8] | controlled_benchmark | 0.021592 | 0.002047 | 0.038884 |
| wall_door_v1 | 36073 | 30014 | 6059 | 0.832035 | internal wall with door opening y in [1.2, 1.8] | phase1_baseline | 0.002381 | 0.000195 | 0.003707 |
| wall_door_v1 | 36073 | 30014 | 6059 | 0.832035 | internal wall with door opening y in [1.2, 1.8] | refinement | 0.021012 | 0.001889 | 0.035888 |
| wall_door_v1 | 36073 | 30014 | 6059 | 0.832035 | internal wall with door opening y in [1.2, 1.8] | refinement_alpha_sweep | 0.021748 | 0.001997 | 0.037939 |

Normalized rates are the main interpretation layer. Raw counts remain in per-profile CSV outputs.
