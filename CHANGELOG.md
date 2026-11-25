# Changelog

## 0.1.0 (November 2026)

First public version.

- Run tab: batch analysis through DeepLabCut's `analyze_videos` in one or more
  worker processes, with the model kept loaded between videos, live progress
  and confidence metrics, retry of a failed video, and optional half precision.
- Auto-tune tab: times batch size, worker count and half precision on the local
  machine and stores the result per GPU, network and frame size.
- Quality tab: per-video scores and advice from the result files.
- Welcome page showing the DeepLabCut version and environment in use, with a
  dialog to change it, and a list of recent projects.
- Checks the git remote for a newer version on start and offers "Update and
  restart", which fast-forwards the checkout and reopens the app.
- The DeepLabCut environment is found without importing DeepLabCut (conda
  environments and common venv folders); the launcher uses the same search and
  asks for a path when nothing is found.
- Works with DeepLabCut 3.0.0rc8 and 3.0.2. On 3.0.2 the app's own frame
  pipeline is not installed because DeepLabCut has one.
- Speed and time left are computed per video, so they no longer drop while a
  worker opens the next video or writes results.
