use cc;
use std::env;
use std::path::Path;

const GENERATE_BINDINGS: bool = false;
const BINDINGS_PATH: &str = "src/cuda/tmp_bindings_cuda.rs";

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // re-run if any changes are made to the kernel code
    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");

    // resolve the location of installed cuda libraries
    let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH is not set");
    let cuda_lib_dir = Path::new(cuda_path.as_str()).join("lib").join("x64");
    let cuda_include_dir = Path::new(cuda_path.as_str()).join("include");

    // add cuda libraries to search path
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    // build the custom cuda kernels and write to host library
    cc::Build::new()
        .cuda(true) // Enable CUDA compilation
        .include(&cuda_include_dir)
        .file("src/cuda/kernels.cu")
        .compile("host"); // Name of the output static library

    // link against custom generated library file
    println!("cargo:rustc-link-lib=static=host");

    // link against cuda libraries
    println!("cargo:rustc-link-lib=dylib=cusolver");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");

    if GENERATE_BINDINGS == true {
        //let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
        let bindings_file = Path::new(BINDINGS_PATH);

        let bindings = bindgen::Builder::default()
            .header("src/cuda/cuda_includes.h")
            .clang_arg(format!("-I{}", cuda_include_dir.display()))
            .generate()
            .expect("Unable to generate cuda bindings!");

        // Write the bindings to the $OUT_DIR/bindings.rs file (to be included in src).
        bindings
            .write_to_file(bindings_file)
            .expect("Couldn't write bindings!");
    }
}
