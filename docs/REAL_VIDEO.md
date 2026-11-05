# Scoring the estimator on real footage

The estimator needs a fixed camera, a calibration (image pixels to ground metres) and manual queue-length labels. Nothing else is required.

## 1. Footage

A fixed camera, an oblique view of a signalised approach with the stop line visible, a few minutes to an hour at 25 or 30 frames per second, no zoom or pan. Footage with trucks that hide the queue is the case the estimator is built for.

## 2. Calibration

Take one frame and read the pixel coordinates of at least six well spread points whose ground positions you can measure (lane-marking corners, stop line ends, crossing stripes), for example with `ffmpeg -ss 5 -i traffic.mp4 -frames:v 1 frame.png` and any image viewer that shows pixel coordinates. Put them in a copy of `configs/site_template.yaml` (`calibration.image_points_px`, `calibration.ground_points_m`, the lanes and the camera position). The command below prints the calibration error in metres.

## 3. Frames to label

```bash
oaqe-label-frames --config configs/site.yaml --video traffic.mp4 --every-s 20 --out-dir label_frames
```

writes one frame per 20 s with the lane centre lines and distance ticks from the stop line, and `label_frames/labels_template.csv` with one empty row per instant and lane. The estimator output is not drawn, so the labels do not depend on it.

## 4. Labels

Fill `queue_length_m` in the template: the distance in metres from the stop line to the rear of the last queued vehicle, read off the ticks (stopped vehicles only; a lane without a queue gets 0). Leave a row empty to skip it. Then

```bash
oaqe-label-frames --finalize label_frames/labels_template.csv --labels-out labels.csv
```

## 5. Estimates and score

```bash
oaqe-video --config configs/site.yaml --video traffic.mp4 --out results/video_estimates.csv
oaqe-score --estimates results/video_estimates.csv --labels labels.csv --out results/real_score.json
```

`real_score.json` holds the number of matched labels, the mean absolute error with a bootstrap interval, the bias and the coverage of the 95 % interval. Report it whatever it shows, with the number of labelled instants and the calibration error.
