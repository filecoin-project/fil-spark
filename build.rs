use std::env;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("--gpu-architecture=sm_86")
        .flag("--generate-code=arch=compute_86,code=sm_86")
        .flag("--generate-code=arch=compute_80,code=sm_80")
        .flag("--generate-code=arch=compute_75,code=sm_75")
        .flag("--threads=0")
        // Without `__ADX__` there will be a linker error on blst when used by downstream
        // dependencies.
        .define("__ADX__", None)
        // `FEATURE_BLS12_281` is needed for Sppark NTT.
        .define("FEATURE_BLS12_381", None)
        .include(env::var_os("DEP_BLST_C_SRC").unwrap())
        .include(env::var_os("DEP_SPPARK_ROOT").unwrap())
        .file("cuda/fil-sppark.cu")
        .compile("fil_sppark");

    println!("cargo:rerun-if-changed=cuda");
}
