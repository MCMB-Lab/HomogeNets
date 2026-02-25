let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = [
    pkgs.uv
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    echo "Sourcing ~/.bashrc..."
    source ~/.bashrc
    uv sync
    source .venv/bin/activate
  '';
}
