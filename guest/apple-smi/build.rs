fn main() {
    let dir = std::env::var("METALPASSTHROUGH_LIB_DIR").unwrap_or_else(|_| "../dist/lib".into());
    println!("cargo:rustc-link-search=native={}", dir);
    println!("cargo:rustc-link-lib=dylib=metalpassthrough");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/metalpassthrough/lib");
    println!("cargo:rerun-if-env-changed=METALPASSTHROUGH_LIB_DIR");
}
