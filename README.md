# SkyMind

Flight dynamics simulations using JAX and PyTorch.

## Project Structure

- `jax/` — JAX flight dynamics simulation with Docker support
- `jax-f16/` — JAX F-16 flight dynamics simulation
- `pytorch/` — PyTorch flight dynamics simulation with Docker support
- `pytorch_warp/` — PyTorch space flight simulation using Warp

## Development Environment Setup (Ubuntu)

Each subproject (`jax/`, `jax-f16/`, `pytorch/`) uses a Nix flake to provide a reproducible development shell. Follow the steps below to get started on Ubuntu.

### 1. Install Nix

Install the Nix package manager using the official installer:

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

After installation, restart your shell

### 2. Enable Flakes, nix-command, and allow-unfree

Create or edit the Nix configuration file:

```bash
mkdir -p ~/.config/nix
cat <<EOF > ~/.config/nix/nix.conf
experimental-features = nix-command flakes
allow-unfree = true
EOF
```

### 3. Enter a Development Shell

Navigate to the subproject you want to work on and start the dev shell:

```bash
cd jax
nix develop
```

This will drop you into a shell with all required dependencies (Python, JAX/PyTorch, etc.) provided by the flake.

The same works for the other subprojects:

```bash
cd pytorch
nix develop
```

### 4. Docker Support (jax, pytorch)

The `jax/` and `pytorch/` subprojects include Docker image builds and helper scripts. Inside the dev shell:

```bash
build-docker   # Build the Docker image
run-docker     # Run the Docker image
```

## License

See [LICENSE](LICENSE).
