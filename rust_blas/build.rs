fn main() {
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
}
