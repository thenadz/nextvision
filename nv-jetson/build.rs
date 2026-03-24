/// Build script for nv-jetson.
///
/// `libnvbufsurface.so` lives under `/usr/lib/aarch64-linux-gnu/tegra/`
/// on Jetson (L4T), which is not in the default linker search path.
/// This is the only platform nv-jetson targets, so we emit the path
/// unconditionally.
fn main() {
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/tegra");
}
