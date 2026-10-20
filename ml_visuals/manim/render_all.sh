#!/usr/bin/env bash
# Render every scene to MP4 (720p30). Output lands in ./media/videos/ml_scenes/720p30/
set -e
cd "$(dirname "$0")"
manim -qm ml_scenes.py GradientDescentScene LinearRegressionScene LogisticBoundaryScene KMeansScene PCAScene OverfittingScene
echo "Done. Now run:  python ../build/build_companion_pdf.py"
