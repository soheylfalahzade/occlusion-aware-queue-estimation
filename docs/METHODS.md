# Methods

Lane coordinates `(s, l)`: `s` is the distance upstream of the stop line, `l` the lateral offset.

## 1. Localisation
The bottom-centre of each bounding box is mapped to the road plane with a homography `H` estimated
by the normalised DLT from `N >= 4` image/ground correspondences. A vehicle's footprint starts at
`s_front = s_anchor + offset`, where the offset depends on the configured anchor (front, centre, rear).

## 2. Stopped vehicles and ghosts
A per-track least-squares velocity over a 2 s window drives a hysteretic state machine
(stopped below `v_stop = 1.0 m/s`, moving above `v_go = 2.0 m/s`). A stopped track that is no longer
detected is kept as a *ghost* for at most 12 s, only while its location is geometrically hidden by
other vehicles; a new track ID within 3 m of a ghost inherits its state (ID-switch re-linking).

## 3. Occlusion model
With the camera at `C` and known vehicle volumes (class-prior length, width, height), a probe point
`p(s) = (s, 0, 0.9 m)` on the lane centre line is *hidden* if the segment `C -> p(s)` intersects the
box of another vehicle (slab test). Probe points inside a visible vehicle count as visible; points
inside a ghost count as hidden.

## 4. Queue observation
Walking upstream over stopped vehicles from the stop line:
1. neighbours with bumper gap `<= 8 m` (first vehicle: `<= 8 m` from the stop line) are chained;
2. a larger gap is bridged only if it contains no *visible* empty stretch longer than 8 m and its hidden
   runs can hold vehicles; each hidden run of length `L` contributes `floor(L / pitch + 0.4)` vehicles
   (`pitch = 4.8 + 2.0 m`), otherwise the chain ends;
3. behind the last member, the contiguous hidden stretch `U` (capped at 30 m) extends the length by
   `p U / 2` with variance `U^2 (p/3 - p^2/4)` (`p = 0.1`);
4. with no stopped vehicle and a hidden stop-line area (`U >= 8 m`) no measurement is produced.

## 5. Filtering and confidence
Length and count are filtered by constant-velocity Kalman filters (white-noise acceleration
`sigma_a = 4 m/s^2` for length, `0.6` for count) with innovation gating: if the normalised innovation
exceeds 9 the measurement variance is inflated by `d^2 / 9`. The 95% interval is `q +/- 1.96 sd`.
`confidence = 1 / (1 + (sd / 3 m)^2) * (0.5 + 0.5 * visible_fraction) * (1 or 0.8 if no measurement)`.

## 6. Reference queue (ground truth)
Vehicles with speed `< 1 m/s` forming a chain from the stop line with gaps `<= 8 m`; the queue length
is the rear bumper of the last member (clipped to the lane length).

## 7. Metrics
MAE/RMSE/bias of queue length, count MAE, empirical coverage and mean width of the 95% interval,
top-1 agreement and relative regret of a max-count lane choice, and MAE by *occlusion rate* (fraction
of true queue members undetected in the last 0.6 s). Intervals: percentile bootstrap over scenarios;
comparisons between variants are paired differences of per-scenario MAE.
