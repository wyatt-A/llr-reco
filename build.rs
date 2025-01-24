// build.rs
use cc;
use std::env;
use std::path::{Path, PathBuf};

//const CUDA_LIB_DIR: &str = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\lib\\x64";
//const CUDA_INCLUDE_DIR: &str = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\include";
fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // resolve the location installed libraries
    let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH is not set");
    let cuda_lib_dir = Path::new(cuda_path.as_str()).join("lib").join("x64");
    let cuda_include_dir = Path::new(cuda_path.as_str()).join("include");

    // cc::Build::new()
    //     .cuda(true) // Enable CUDA compilation
    //     .include("src/cuda_kernels")
    //     .file("src/cuda_kernels/diag_mm_batched.cu") // Path to your CUDA file
    //     .compile("libkernels"); // Name of the output static library
    //
    // cc::Build::new()
    //     .file("src/cuda_kernels/diag_mm_batched_exec.c")
    //     .include("src/cuda_kernels")
    //     .compile("libhost");

    // let cuda_file =
    //
    // let status = Command::new("nvcc")
    //     .args(&[
    //         cuda_file,                  // The input CUDA file
    //         "-lib",                     // Generate a static library
    //         "-arch=sm_60",              // GPU architecture (adjust as needed)
    //         "-std=c++17",               // Use the C++17 standard
    //         "-o", &output_lib,          // Output static library path
    //     ])
    //     .status()
    //     .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed!");
    }

    // --- 1. Instruct Cargo to link against cusolver, cublas, cudart, etc. ---
    // On Windows, these are typically .lib files, but at runtime you'll need
    // the corresponding .dll to be in your PATH or alongside your executable.
    println!("cargo:rustc-link-lib=static=kernels"); // Link the custom static CUDA library
    println!("cargo:rustc-link-lib=static=host"); // Link the custom static host library
    println!("cargo:rustc-link-lib=dylib=cusolver");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");

    // If needed, add the directory where the .lib files live.
    // Update this path to match your CUDA version and install path:
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    // --- 2. Use bindgen to generate Rust bindings ---

    // Where to place the generated bindings.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    //println!("{:?}", out_path);
    //panic!("stop!");
    // Configure bindgen
    let bindings = bindgen::Builder::default()
        // Tell bindgen which header(s) to parse.
        .header("wrapper.h")
        // Ensure bindgen sees the same include path as the compiler would.
        //.clang_arg("-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
        .clang_arg(format!("-I{}", cuda_include_dir.display()))
        // `parse_callbacks` tells bindgen to generate Cargo directives, so
        // it re-runs build scripts automatically when the wrapper or included
        // files change.
        //.parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Generate the actual bindings.
        .generate()
        .expect("Unable to generate cusolver bindings!");

    // Write the bindings to the $OUT_DIR/bindings.rs file (to be included in src).
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");
}
