# Run with `nix-shell cuda-shell.nix`
{ pkgs ? import <nixos> {
   config = {
      allowUnfree = true;
      cudaSupport = true;
   };
} }:
pkgs.gcc13Stdenv.mkDerivation {
   name = "cuda-env-shell";
   buildInputs = with pkgs; [
     git gitRepo gnupg autoconf curl
     procps gnumake util-linux m4 gperf unzip
     cudatoolkit linuxPackages_6_6.nvidia_x11
     libGLU libGL gcc13 gcc13Stdenv.cc
     xorg.libXi xorg.libXmu freeglut
     xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
     ncurses5 binutils
     clang
     llvmPackages_17.bintools
     rustup
     cmake
     libtorch-bin
   ]
   ++ gcc13Stdenv.defaultBuildInputs
   ++ gcc13Stdenv.defaultNativeBuildInputs;
   RUSTC_VERSION = pkgs.lib.readFile ./rust-toolchain;
   # https://github.com/rust-lang/rust-bindgen#environment-variables
   LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
   shellHook = ''
      export PATH=$PATH:''${CARGO_HOME:-~/.cargo}/bin
      export PATH=$PATH:''${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
      export CUDA_PATH=${pkgs.cudatoolkit}
      export LD_LIBRARY_PATH=${pkgs.libtorch-bin}/lib:${pkgs.linuxPackages_6_6.nvidia_x11}/lib:${pkgs.ncurses5}/lib:${pkgs.gcc13Stdenv.cc.cc.lib}:$LD_LIBRARY_PATH
      export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages_6_6.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
      export LIBTORCH=${pkgs.libtorch-bin}
      export LIBTORCH_LIB=${pkgs.libtorch-bin}
      export LIBTORCH_INCLUDE=${pkgs.libtorch-bin.dev}
      export TORCH_CUDA_VERSION=cu118
   '';
   # export LIBTORCH=${pkgs.python310Packages.torchWithCuda}/lib/python3.10/site-packages/torch
   # Add precompiled library to rustc search path
   RUSTFLAGS = (builtins.map (a: ''-L ${a}/lib'') [
   # add libraries here (e.g. pkgs.libvmi)
   ]);
   # Add glibc, clang, glib and other headers to bindgen search path
   BINDGEN_EXTRA_CLANG_ARGS = 
   # Includes with normal include path
   (builtins.map (a: ''-I"${a}/include"'') [
      # add dev libraries here (e.g. pkgs.libvmi.dev)
      pkgs.gcc13.libc.dev 
   ])
   ++ [
      ''-I"${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${pkgs.llvmPackages_latest.libclang.version}/include"''
      ''-I"${pkgs.glib.dev}/include/glib-2.0"''
      ''-I${pkgs.glib.out}/lib/glib-2.0/include/''
   ];
}
