# Large-scale scripts

Command-line entry points of the large-scale analysis, run from the repository root:

| Script | What |
|---|---|
| `run_analysis.py` | the analysis itself (`results/`, `figures/`); `--help` lists the options |
| `co_nc_sweep.py` | ranks equal-size CO/NC design pairs by coverage ratio and plots the win rate over the top pairs (`figures/winrate_vs_npairs.png`) |
| `make_surface_co_nc.py`, `build_surface_co_nc.sh` | the CO-versus-NC fitted-surface figure (`figures/surface_co_vs_nc.png`), drawn with the paper's surface code |
| `check_readme_math.py` | checks that the README's math renders on GitHub |
