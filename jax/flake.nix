{
  description = "JAX flight dynamics simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: [
        ps.jax
        ps.jaxlib
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
      };
    };
}
