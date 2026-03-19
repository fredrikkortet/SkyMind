{
  description = "pytorch flight dynamics simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: [
        ps.torch
      ]);
      appSrc = pkgs.runCommand "app-src" { } ''
        mkdir -p $out
        cp -r ${./src} $out/app
      '';
      dockerImage = pkgs.dockerTools.buildImage {
        name = "pytorch-sim";
        tag = "latest";
        copyToRoot = pkgs.buildEnv {
          name = "image-root";
          paths = [
            python
            pkgs.bashInteractive
            pkgs.coreutils
            appSrc
          ];
          pathsToLink = [
            "/bin"
            "/lib"
            "/app"
          ];
        };
        config = {
          Cmd = [
            "${python}/bin/python"
            "pytorch_sim.py"
          ];
          WorkingDir = "/app";
        };
      };
      build-docker = pkgs.writeShellScriptBin "build-docker" ''
        set -e
        echo "Building Docker image..."
        nix build .#dockerImage
        echo "Loading into Docker..."
        docker load < result
        echo "Done! Run with: run-docker"
      '';
      run-docker = pkgs.writeShellScriptBin "run-docker" ''
        set -e
        echo "Running Docker image..."
        docker run -it pytorch-sim:latest
        echo "Done running Docker image!"
      '';
    in
    {
      packages.${system}.dockerImage = dockerImage;

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          build-docker
          run-docker
          pkgs.docker
        ];
        shellHook = ''
          if ! docker info &>/dev/null 2>&1; then
            echo "Starting Docker daemon..."
            sudo sh -c "$(which dockerd) -G '$(id -gn)' > /dev/null 2>&1 < /dev/null &"
            while ! docker info &>/dev/null 2>&1; do
              sleep 1
            done
            echo "Docker daemon started."
          fi
        '';
      };
    };
}
