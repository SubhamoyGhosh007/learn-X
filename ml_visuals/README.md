# ML from Zero to Hero: extras bundle

The main deliverable is the complete report PDF (ML_Zero_to_Hero_Complete.pdf, all text + all 42 figures inline).
This bundle holds everything needed to re-render or extend the figures.

concepts/gifs/          28 real animations (GIF), one per math idea / algorithm
concepts/diagrams/      28 Graphviz diagrams (.dot source, .png, .svg)
concepts/sheets/        diagram + animation frames composed per topic (what the report embeds)
concepts/make_*.py      scripts that regenerate all of the above
graphviz/               8 earlier diagrams (workflow, taxonomy, chooser, CV, ensembles, tree, GD loop, pipeline)
storyboards/            filmstrips of the six original animated scenes
manim/ml_scenes.py      Manim Community scenes for those six (not rendered here; run manim/render_all.sh)
build/                  build_companion_pdf.py (swaps in real Manim frames after you render)

Regenerate the concept material:
  cd concepts && python make_diagrams.py && python make_anims.py && python make_sheets.py
Render Manim:  pip install manim && cd manim && ./render_all.sh
