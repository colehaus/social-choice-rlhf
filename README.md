# Social choice RLHF

This repo contains code for an alternative RLHF reward modeling setup from a social choice perspective. There's an accompanying [blog post](https://col-ex.org/posts/social-choice-rlhf/).

## Usage

As far as dependencies and setup, there are a couple of options:

- There's a `pyproject.toml` that contains Poetry declarations, but I never use `poetry` directly so I can't guarantee its correctness.
- I use Poetry indirectly via `poetry2nix`. If you're a Nix user, invoking `nix develop` should get you a CPU-only setup.
