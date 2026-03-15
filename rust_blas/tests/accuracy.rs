//! Accuracy tests — compare Rust BLAS against the reference Fortran BLAS
//! (system libblas.so linked via build.rs) at multiple problem sizes.
//!
//! Routines tested: all benchmarked ones, both f32 and f64, at n ∈ {64, 512, 2048}.
//! Results written to /tmp/accuracy_results.json.
use rust_blas::level1;
use rust_blas::level2;
use rust_blas::level3;
use rust_blas::types::{Diag, Side, Trans, Uplo};
use std::io::Write;

// ── Fortran BLAS FFI — f64 ────────────────────────────────────────────────
extern "C" {
    fn dasum_(n: *const i32, dx: *const f64, incx: *const i32) -> f64;
    fn daxpy_(n: *const i32, da: *const f64, dx: *const f64, incx: *const i32, dy: *mut f64, incy: *const i32);
    fn ddot_(n: *const i32, dx: *const f64, incx: *const i32, dy: *const f64, incy: *const i32) -> f64;
    fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn dscal_(n: *const i32, da: *const f64, dx: *mut f64, incx: *const i32);
    fn dswap_(n: *const i32, dx: *mut f64, incx: *const i32, dy: *mut f64, incy: *const i32);
    fn dgemv_(trans: *const u8, m: *const i32, n: *const i32, alpha: *const f64,
              a: *const f64, lda: *const i32, x: *const f64, incx: *const i32,
              beta: *const f64, y: *mut f64, incy: *const i32);
    fn dger_(m: *const i32, n: *const i32, alpha: *const f64,
             x: *const f64, incx: *const i32, y: *const f64, incy: *const i32,
             a: *mut f64, lda: *const i32);
    fn dsymv_(uplo: *const u8, n: *const i32, alpha: *const f64,
              a: *const f64, lda: *const i32, x: *const f64, incx: *const i32,
              beta: *const f64, y: *mut f64, incy: *const i32);
    fn dtrmv_(uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
              a: *const f64, lda: *const i32, x: *mut f64, incx: *const i32);
    fn dtrsv_(uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
              a: *const f64, lda: *const i32, x: *mut f64, incx: *const i32);
    fn dgemm_(transa: *const u8, transb: *const u8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32,
              b: *const f64, ldb: *const i32, beta: *const f64,
              c: *mut f64, ldc: *const i32);
    fn dsymm_(side: *const u8, uplo: *const u8, m: *const i32, n: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32,
              b: *const f64, ldb: *const i32, beta: *const f64,
              c: *mut f64, ldc: *const i32);
    fn dsyrk_(uplo: *const u8, trans: *const u8, n: *const i32, k: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32,
              beta: *const f64, c: *mut f64, ldc: *const i32);
    fn dtrmm_(side: *const u8, uplo: *const u8, transa: *const u8, diag: *const u8,
              m: *const i32, n: *const i32, alpha: *const f64,
              a: *const f64, lda: *const i32, b: *mut f64, ldb: *const i32);
    fn dtrsm_(side: *const u8, uplo: *const u8, transa: *const u8, diag: *const u8,
              m: *const i32, n: *const i32, alpha: *const f64,
              a: *const f64, lda: *const i32, b: *mut f64, ldb: *const i32);
}

// ── Fortran BLAS FFI — f32 ────────────────────────────────────────────────
extern "C" {
    fn sasum_(n: *const i32, sx: *const f32, incx: *const i32) -> f32;
    fn saxpy_(n: *const i32, sa: *const f32, sx: *const f32, incx: *const i32, sy: *mut f32, incy: *const i32);
    fn sdot_(n: *const i32, sx: *const f32, incx: *const i32, sy: *const f32, incy: *const i32) -> f32;
    fn snrm2_(n: *const i32, x: *const f32, incx: *const i32) -> f32;
    fn sscal_(n: *const i32, sa: *const f32, sx: *mut f32, incx: *const i32);
    fn sswap_(n: *const i32, sx: *mut f32, incx: *const i32, sy: *mut f32, incy: *const i32);
    fn sgemv_(trans: *const u8, m: *const i32, n: *const i32, alpha: *const f32,
              a: *const f32, lda: *const i32, x: *const f32, incx: *const i32,
              beta: *const f32, y: *mut f32, incy: *const i32);
    fn sger_(m: *const i32, n: *const i32, alpha: *const f32,
             x: *const f32, incx: *const i32, y: *const f32, incy: *const i32,
             a: *mut f32, lda: *const i32);
    fn ssymv_(uplo: *const u8, n: *const i32, alpha: *const f32,
              a: *const f32, lda: *const i32, x: *const f32, incx: *const i32,
              beta: *const f32, y: *mut f32, incy: *const i32);
    fn strmv_(uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
              a: *const f32, lda: *const i32, x: *mut f32, incx: *const i32);
    fn strsv_(uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
              a: *const f32, lda: *const i32, x: *mut f32, incx: *const i32);
    fn sgemm_(transa: *const u8, transb: *const u8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const f32, a: *const f32, lda: *const i32,
              b: *const f32, ldb: *const i32, beta: *const f32,
              c: *mut f32, ldc: *const i32);
    fn ssymm_(side: *const u8, uplo: *const u8, m: *const i32, n: *const i32,
              alpha: *const f32, a: *const f32, lda: *const i32,
              b: *const f32, ldb: *const i32, beta: *const f32,
              c: *mut f32, ldc: *const i32);
    fn ssyrk_(uplo: *const u8, trans: *const u8, n: *const i32, k: *const i32,
              alpha: *const f32, a: *const f32, lda: *const i32,
              beta: *const f32, c: *mut f32, ldc: *const i32);
    fn strmm_(side: *const u8, uplo: *const u8, transa: *const u8, diag: *const u8,
              m: *const i32, n: *const i32, alpha: *const f32,
              a: *const f32, lda: *const i32, b: *mut f32, ldb: *const i32);
    fn strsm_(side: *const u8, uplo: *const u8, transa: *const u8, diag: *const u8,
              m: *const i32, n: *const i32, alpha: *const f32,
              a: *const f32, lda: *const i32, b: *mut f32, ldb: *const i32);
}

// ── Helpers: f64 ─────────────────────────────────────────────────────────

fn make_vec(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64) / (n as f64)).collect()
}
fn make_mat(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|k| {
        let i = (k % rows) + 1;
        let j = (k / rows) + 1;
        ((i * j) as f64 * 0.1).sin()
    }).collect()
}
/// Well-conditioned upper-triangular: large diagonal, strict upper entries, zeros below.
fn make_tri(n: usize) -> Vec<f64> {
    let mut a = make_mat(n, n);
    for j in 0..n {
        for i in (j + 1)..n { a[j * n + i] = 0.0; }
        a[j * n + j] = n as f64 + j as f64 + 1.0;
    }
    a
}
fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x.abs().max(y.abs()).max(1e-300);
        (x - y).abs() / d
    }).fold(0.0_f64, f64::max)
}
fn max_rel_err_upper(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut max_e = 0.0_f64;
    for j in 0..n {
        for i in 0..=j {
            let x = a[j * n + i];
            let y = b[j * n + i];
            let d = x.abs().max(y.abs()).max(1e-300);
            let e = (x - y).abs() / d;
            if e > max_e { max_e = e; }
        }
    }
    max_e
}

// ── Helpers: f32 ─────────────────────────────────────────────────────────

fn make_vec_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i + 1) as f32) / (n as f32)).collect()
}
fn make_mat_f32(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|k| {
        let i = (k % rows) + 1;
        let j = (k / rows) + 1;
        ((i * j) as f32 * 0.1_f32).sin()
    }).collect()
}
fn make_tri_f32(n: usize) -> Vec<f32> {
    let mut a = make_mat_f32(n, n);
    for j in 0..n {
        for i in (j + 1)..n { a[j * n + i] = 0.0_f32; }
        a[j * n + j] = n as f32 + j as f32 + 1.0_f32;
    }
    a
}
fn max_rel_err_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = (x.abs() as f64).max(y.abs() as f64).max(1e-37_f64);
        ((x - y).abs() as f64) / d
    }).fold(0.0_f64, f64::max)
}
fn max_rel_err_upper_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    let mut max_e = 0.0_f64;
    for j in 0..n {
        for i in 0..=j {
            let x = a[j * n + i] as f64;
            let y = b[j * n + i] as f64;
            let d = x.abs().max(y.abs()).max(1e-37_f64);
            let e = (x - y).abs() / d;
            if e > max_e { max_e = e; }
        }
    }
    max_e
}

// ── Accuracy thresholds & result struct ──────────────────────────────────

/// f64: ≤1e-10. That is ~450× machine epsilon × n=2048, so should pass comfortably
/// for all implemented routines that mirror the reference algorithm.
const EPS_F64: f64 = 1e-10;
/// f32: ≤5e-4. Single-precision unit roundoff ~1.2e-7; for n=2048 accumulation
/// worst case ~2.5e-4. This threshold catches real implementation bugs.
const EPS_F32: f64 = 5e-4;

struct AccResult {
    routine: String,
    precision: String,
    n: usize,
    max_rel_error: f64,
    threshold: f64,
    passed: bool,
}

impl AccResult {
    fn f64(routine: &str, n: usize, err: f64) -> Self {
        AccResult { routine: routine.into(), precision: "f64".into(), n,
                    max_rel_error: err, threshold: EPS_F64, passed: err <= EPS_F64 }
    }
    fn f32(routine: &str, n: usize, err: f64) -> Self {
        AccResult { routine: routine.into(), precision: "f32".into(), n,
                    max_rel_error: err, threshold: EPS_F32, passed: err <= EPS_F32 }
    }
}

// ── f64 test functions (parameterised by n) ───────────────────────────────

fn t_f64_asum(n: usize) -> AccResult {
    let x = make_vec(n);
    let r = unsafe { dasum_(&(n as i32), x.as_ptr(), &1) };
    let v = level1::asum(n, &x, 1);
    AccResult::f64("asum", n, (r - v).abs() / r.abs().max(1e-300))
}
fn t_f64_axpy(n: usize) -> AccResult {
    let x = make_vec(n); let y0 = make_vec(n); let a = 2.5_f64;
    let mut yr = y0.clone();
    unsafe { daxpy_(&(n as i32), &a, x.as_ptr(), &1, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone(); level1::axpy(n, a, &x, 1, &mut yv, 1);
    AccResult::f64("axpy", n, max_rel_err(&yr, &yv))
}
fn t_f64_dot(n: usize) -> AccResult {
    let x = make_vec(n); let y = make_vec(n);
    let r = unsafe { ddot_(&(n as i32), x.as_ptr(), &1, y.as_ptr(), &1) };
    let v = level1::dot(n, &x, 1, &y, 1);
    AccResult::f64("dot", n, (r - v).abs() / r.abs().max(1e-300))
}
fn t_f64_nrm2(n: usize) -> AccResult {
    let x = make_vec(n);
    let r = unsafe { dnrm2_(&(n as i32), x.as_ptr(), &1) };
    let v = level1::nrm2(n, &x, 1);
    AccResult::f64("nrm2", n, (r - v).abs() / r.abs().max(1e-300))
}
fn t_f64_scal(n: usize) -> AccResult {
    let x0 = make_vec(n); let a = 3.14_f64;
    let mut xr = x0.clone(); unsafe { dscal_(&(n as i32), &a, xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone(); level1::scal(n, a, &mut xv, 1);
    AccResult::f64("scal", n, max_rel_err(&xr, &xv))
}
fn t_f64_swap(n: usize) -> AccResult {
    let x0 = make_vec(n);
    let y0: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
    let (mut xr, mut yr) = (x0.clone(), y0.clone());
    unsafe { dswap_(&(n as i32), xr.as_mut_ptr(), &1, yr.as_mut_ptr(), &1) };
    let (mut xv, mut yv) = (x0.clone(), y0.clone());
    level1::swap(n, &mut xv, 1, &mut yv, 1);
    AccResult::f64("swap", n, max_rel_err(&xr, &xv).max(max_rel_err(&yr, &yv)))
}
fn t_f64_gemv(n: usize) -> AccResult {
    let a = make_mat(n, n); let x = make_vec(n); let y0 = make_vec(n);
    let (al, be) = (1.5_f64, 0.5_f64);
    let mut yr = y0.clone();
    unsafe { dgemv_(b"N".as_ptr(), &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    x.as_ptr(), &1, &be, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone();
    level2::gemv(Trans::No, n, n, al, &a, n, &x, 1, be, &mut yv, 1);
    AccResult::f64("gemv", n, max_rel_err(&yr, &yv))
}
fn t_f64_ger(n: usize) -> AccResult {
    let x = make_vec(n); let y = make_vec(n); let a0 = make_mat(n, n); let al = 0.7_f64;
    let mut ar = a0.clone();
    unsafe { dger_(&(n as i32), &(n as i32), &al, x.as_ptr(), &1, y.as_ptr(), &1,
                   ar.as_mut_ptr(), &(n as i32)) };
    let mut av = a0.clone();
    level2::ger(n, n, al, &x, 1, &y, 1, &mut av, n);
    AccResult::f64("ger", n, max_rel_err(&ar, &av))
}
fn t_f64_symv(n: usize) -> AccResult {
    let a = make_mat(n, n); let x = make_vec(n); let y0 = make_vec(n);
    let (al, be) = (1.2_f64, 0.3_f64);
    let mut yr = y0.clone();
    unsafe { dsymv_(b"U".as_ptr(), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    x.as_ptr(), &1, &be, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone();
    level2::symv(Uplo::Upper, n, al, &a, n, &x, 1, be, &mut yv, 1);
    AccResult::f64("symv", n, max_rel_err(&yr, &yv))
}
fn t_f64_trmv(n: usize) -> AccResult {
    let a = make_tri(n); let x0 = make_vec(n);
    let mut xr = x0.clone();
    unsafe { dtrmv_(b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(), &(n as i32),
                    a.as_ptr(), &(n as i32), xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone();
    level2::trmv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut xv, 1);
    AccResult::f64("trmv", n, max_rel_err(&xr, &xv))
}
fn t_f64_trsv(n: usize) -> AccResult {
    let a = make_tri(n); let x0 = make_vec(n);
    let mut xr = x0.clone();
    unsafe { dtrsv_(b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(), &(n as i32),
                    a.as_ptr(), &(n as i32), xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone();
    level2::trsv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut xv, 1);
    AccResult::f64("trsv", n, max_rel_err(&xr, &xv))
}
fn t_f64_gemm(n: usize) -> AccResult {
    let a = make_mat(n, n); let b = make_mat(n, n); let c0 = make_mat(n, n);
    let (al, be) = (1.0_f64, 0.0_f64);
    let mut cr = c0.clone();
    unsafe { dgemm_(b"N".as_ptr(), b"N".as_ptr(), &(n as i32), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), b.as_ptr(), &(n as i32),
                    &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::gemm(Trans::No, Trans::No, n, n, n, al, &a, n, &b, n, be, &mut cv, n);
    AccResult::f64("gemm", n, max_rel_err(&cr, &cv))
}
fn t_f64_symm(n: usize) -> AccResult {
    let a = make_mat(n, n); let b = make_mat(n, n); let c0 = make_mat(n, n);
    let (al, be) = (1.0_f64, 0.0_f64);
    let mut cr = c0.clone();
    unsafe { dsymm_(b"L".as_ptr(), b"U".as_ptr(), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), b.as_ptr(), &(n as i32),
                    &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::symm(Side::Left, Uplo::Upper, n, n, al, &a, n, &b, n, be, &mut cv, n);
    AccResult::f64("symm", n, max_rel_err(&cr, &cv))
}
fn t_f64_syrk(n: usize) -> AccResult {
    let a = make_mat(n, n); let c0 = make_mat(n, n);
    let (al, be) = (1.0_f64, 0.0_f64);
    let mut cr = c0.clone();
    unsafe { dsyrk_(b"U".as_ptr(), b"N".as_ptr(), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::syrk(Uplo::Upper, Trans::No, n, n, al, &a, n, be, &mut cv, n);
    AccResult::f64("syrk", n, max_rel_err_upper(&cr, &cv, n))
}
fn t_f64_trmm(n: usize) -> AccResult {
    let a = make_tri(n); let b0 = make_mat(n, n); let al = 1.0_f64;
    let mut br = b0.clone();
    unsafe { dtrmm_(b"L".as_ptr(), b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(),
                    &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    br.as_mut_ptr(), &(n as i32)) };
    let mut bv = b0.clone();
    level3::trmm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, n, n, al, &a, n, &mut bv, n);
    AccResult::f64("trmm", n, max_rel_err(&br, &bv))
}
fn t_f64_trsm(n: usize) -> AccResult {
    let a = make_tri(n); let b0 = make_mat(n, n); let al = 1.0_f64;
    let mut br = b0.clone();
    unsafe { dtrsm_(b"L".as_ptr(), b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(),
                    &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    br.as_mut_ptr(), &(n as i32)) };
    let mut bv = b0.clone();
    level3::trsm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, n, n, al, &a, n, &mut bv, n);
    AccResult::f64("trsm", n, max_rel_err(&br, &bv))
}

// ── f32 test functions (parameterised by n) ───────────────────────────────

fn t_f32_asum(n: usize) -> AccResult {
    let x = make_vec_f32(n);
    let r = unsafe { sasum_(&(n as i32), x.as_ptr(), &1) };
    let v = level1::asum(n, &x, 1);
    let err = ((r - v).abs() as f64) / (r.abs() as f64).max(1e-37);
    AccResult::f32("asum", n, err)
}
fn t_f32_axpy(n: usize) -> AccResult {
    let x = make_vec_f32(n); let y0 = make_vec_f32(n); let a = 2.5_f32;
    let mut yr = y0.clone();
    unsafe { saxpy_(&(n as i32), &a, x.as_ptr(), &1, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone(); level1::axpy(n, a, &x, 1, &mut yv, 1);
    AccResult::f32("axpy", n, max_rel_err_f32(&yr, &yv))
}
fn t_f32_dot(n: usize) -> AccResult {
    let x = make_vec_f32(n); let y = make_vec_f32(n);
    let r = unsafe { sdot_(&(n as i32), x.as_ptr(), &1, y.as_ptr(), &1) };
    let v = level1::dot(n, &x, 1, &y, 1);
    let err = ((r - v).abs() as f64) / (r.abs() as f64).max(1e-37);
    AccResult::f32("dot", n, err)
}
fn t_f32_nrm2(n: usize) -> AccResult {
    let x = make_vec_f32(n);
    let r = unsafe { snrm2_(&(n as i32), x.as_ptr(), &1) };
    let v = level1::nrm2(n, &x, 1);
    let err = ((r - v).abs() as f64) / (r.abs() as f64).max(1e-37);
    AccResult::f32("nrm2", n, err)
}
fn t_f32_scal(n: usize) -> AccResult {
    let x0 = make_vec_f32(n); let a = 3.14_f32;
    let mut xr = x0.clone(); unsafe { sscal_(&(n as i32), &a, xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone(); level1::scal(n, a, &mut xv, 1);
    AccResult::f32("scal", n, max_rel_err_f32(&xr, &xv))
}
fn t_f32_swap(n: usize) -> AccResult {
    let x0 = make_vec_f32(n);
    let y0: Vec<f32> = (0..n).map(|i| i as f32 * 0.3_f32).collect();
    let (mut xr, mut yr) = (x0.clone(), y0.clone());
    unsafe { sswap_(&(n as i32), xr.as_mut_ptr(), &1, yr.as_mut_ptr(), &1) };
    let (mut xv, mut yv) = (x0.clone(), y0.clone());
    level1::swap(n, &mut xv, 1, &mut yv, 1);
    AccResult::f32("swap", n, max_rel_err_f32(&xr, &xv).max(max_rel_err_f32(&yr, &yv)))
}
fn t_f32_gemv(n: usize) -> AccResult {
    let a = make_mat_f32(n, n); let x = make_vec_f32(n); let y0 = make_vec_f32(n);
    let (al, be) = (1.5_f32, 0.5_f32);
    let mut yr = y0.clone();
    unsafe { sgemv_(b"N".as_ptr(), &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    x.as_ptr(), &1, &be, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone();
    level2::gemv(Trans::No, n, n, al, &a, n, &x, 1, be, &mut yv, 1);
    AccResult::f32("gemv", n, max_rel_err_f32(&yr, &yv))
}
fn t_f32_ger(n: usize) -> AccResult {
    let x = make_vec_f32(n); let y = make_vec_f32(n); let a0 = make_mat_f32(n, n); let al = 0.7_f32;
    let mut ar = a0.clone();
    unsafe { sger_(&(n as i32), &(n as i32), &al, x.as_ptr(), &1, y.as_ptr(), &1,
                   ar.as_mut_ptr(), &(n as i32)) };
    let mut av = a0.clone();
    level2::ger(n, n, al, &x, 1, &y, 1, &mut av, n);
    AccResult::f32("ger", n, max_rel_err_f32(&ar, &av))
}
fn t_f32_symv(n: usize) -> AccResult {
    let a = make_mat_f32(n, n); let x = make_vec_f32(n); let y0 = make_vec_f32(n);
    let (al, be) = (1.2_f32, 0.3_f32);
    let mut yr = y0.clone();
    unsafe { ssymv_(b"U".as_ptr(), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    x.as_ptr(), &1, &be, yr.as_mut_ptr(), &1) };
    let mut yv = y0.clone();
    level2::symv(Uplo::Upper, n, al, &a, n, &x, 1, be, &mut yv, 1);
    AccResult::f32("symv", n, max_rel_err_f32(&yr, &yv))
}
fn t_f32_trmv(n: usize) -> AccResult {
    let a = make_tri_f32(n); let x0 = make_vec_f32(n);
    let mut xr = x0.clone();
    unsafe { strmv_(b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(), &(n as i32),
                    a.as_ptr(), &(n as i32), xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone();
    level2::trmv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut xv, 1);
    AccResult::f32("trmv", n, max_rel_err_f32(&xr, &xv))
}
fn t_f32_trsv(n: usize) -> AccResult {
    let a = make_tri_f32(n); let x0 = make_vec_f32(n);
    let mut xr = x0.clone();
    unsafe { strsv_(b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(), &(n as i32),
                    a.as_ptr(), &(n as i32), xr.as_mut_ptr(), &1) };
    let mut xv = x0.clone();
    level2::trsv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut xv, 1);
    AccResult::f32("trsv", n, max_rel_err_f32(&xr, &xv))
}
fn t_f32_gemm(n: usize) -> AccResult {
    let a = make_mat_f32(n, n); let b = make_mat_f32(n, n); let c0 = make_mat_f32(n, n);
    let (al, be) = (1.0_f32, 0.0_f32);
    let mut cr = c0.clone();
    unsafe { sgemm_(b"N".as_ptr(), b"N".as_ptr(), &(n as i32), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), b.as_ptr(), &(n as i32),
                    &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::gemm(Trans::No, Trans::No, n, n, n, al, &a, n, &b, n, be, &mut cv, n);
    AccResult::f32("gemm", n, max_rel_err_f32(&cr, &cv))
}
fn t_f32_symm(n: usize) -> AccResult {
    let a = make_mat_f32(n, n); let b = make_mat_f32(n, n); let c0 = make_mat_f32(n, n);
    let (al, be) = (1.0_f32, 0.0_f32);
    let mut cr = c0.clone();
    unsafe { ssymm_(b"L".as_ptr(), b"U".as_ptr(), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), b.as_ptr(), &(n as i32),
                    &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::symm(Side::Left, Uplo::Upper, n, n, al, &a, n, &b, n, be, &mut cv, n);
    AccResult::f32("symm", n, max_rel_err_f32(&cr, &cv))
}
fn t_f32_syrk(n: usize) -> AccResult {
    let a = make_mat_f32(n, n); let c0 = make_mat_f32(n, n);
    let (al, be) = (1.0_f32, 0.0_f32);
    let mut cr = c0.clone();
    unsafe { ssyrk_(b"U".as_ptr(), b"N".as_ptr(), &(n as i32), &(n as i32),
                    &al, a.as_ptr(), &(n as i32), &be, cr.as_mut_ptr(), &(n as i32)) };
    let mut cv = c0.clone();
    level3::syrk(Uplo::Upper, Trans::No, n, n, al, &a, n, be, &mut cv, n);
    AccResult::f32("syrk", n, max_rel_err_upper_f32(&cr, &cv, n))
}
fn t_f32_trmm(n: usize) -> AccResult {
    let a = make_tri_f32(n); let b0 = make_mat_f32(n, n); let al = 1.0_f32;
    let mut br = b0.clone();
    unsafe { strmm_(b"L".as_ptr(), b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(),
                    &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    br.as_mut_ptr(), &(n as i32)) };
    let mut bv = b0.clone();
    level3::trmm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, n, n, al, &a, n, &mut bv, n);
    AccResult::f32("trmm", n, max_rel_err_f32(&br, &bv))
}
fn t_f32_trsm(n: usize) -> AccResult {
    let a = make_tri_f32(n); let b0 = make_mat_f32(n, n); let al = 1.0_f32;
    let mut br = b0.clone();
    unsafe { strsm_(b"L".as_ptr(), b"U".as_ptr(), b"N".as_ptr(), b"N".as_ptr(),
                    &(n as i32), &(n as i32), &al, a.as_ptr(), &(n as i32),
                    br.as_mut_ptr(), &(n as i32)) };
    let mut bv = b0.clone();
    level3::trsm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, n, n, al, &a, n, &mut bv, n);
    AccResult::f32("trsm", n, max_rel_err_f32(&br, &bv))
}

// ── Main test ─────────────────────────────────────────────────────────────

const BENCH_SIZES: &[usize] = &[64, 512, 2048];

#[test]
fn write_accuracy_report() {
    let mut results: Vec<AccResult> = Vec::new();

    for &n in BENCH_SIZES {
        // f64 suite
        results.push(t_f64_asum(n));
        results.push(t_f64_axpy(n));
        results.push(t_f64_dot(n));
        results.push(t_f64_nrm2(n));
        results.push(t_f64_scal(n));
        results.push(t_f64_swap(n));
        results.push(t_f64_gemv(n));
        results.push(t_f64_ger(n));
        results.push(t_f64_symv(n));
        results.push(t_f64_trmv(n));
        results.push(t_f64_trsv(n));
        results.push(t_f64_gemm(n));
        results.push(t_f64_symm(n));
        results.push(t_f64_syrk(n));
        results.push(t_f64_trmm(n));
        results.push(t_f64_trsm(n));
        // f32 suite
        results.push(t_f32_asum(n));
        results.push(t_f32_axpy(n));
        results.push(t_f32_dot(n));
        results.push(t_f32_nrm2(n));
        results.push(t_f32_scal(n));
        results.push(t_f32_swap(n));
        results.push(t_f32_gemv(n));
        results.push(t_f32_ger(n));
        results.push(t_f32_symv(n));
        results.push(t_f32_trmv(n));
        results.push(t_f32_trsv(n));
        results.push(t_f32_gemm(n));
        results.push(t_f32_symm(n));
        results.push(t_f32_syrk(n));
        results.push(t_f32_trmm(n));
        results.push(t_f32_trsm(n));
    }

    // Pretty-print to stdout
    println!("\n{:=<74}", "");
    println!("{:^74}", "Accuracy — Rust vs. Reference Fortran BLAS");
    println!("{:=<74}", "");
    println!("{:<8} {:<5} {:>6}  {:>12}  {:>10}  {}", "Routine", "Prec", "N", "Max Rel Err", "Threshold", "Status");
    println!("{:-<74}", "");
    let mut all_pass = true;
    for r in &results {
        let status = if r.passed { "PASS ✓" } else { "FAIL ✗" };
        println!("{:<8} {:<5} {:>6}  {:>12.3e}  {:>10.1e}  {}",
                 r.routine, r.precision, r.n, r.max_rel_error, r.threshold, status);
        if !r.passed { all_pass = false; }
    }
    println!("{:=<74}", "");
    let n_pass = results.iter().filter(|r| r.passed).count();
    let n_fail = results.len() - n_pass;
    println!("Total: {} tests — {} PASSED, {} FAILED", results.len(), n_pass, n_fail);

    // Write JSON
    let mut json = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        json.push_str(&format!(
            "  {{\"routine\": \"{}\", \"precision\": \"{}\", \"n\": {}, \
             \"max_rel_error\": {:.6e}, \"threshold\": {:.1e}, \"passed\": {}}}{}\n",
            r.routine, r.precision, r.n, r.max_rel_error, r.threshold, r.passed, comma
        ));
    }
    json.push_str("]\n");

    let path = "/tmp/accuracy_results.json";
    let mut f = std::fs::File::create(path).expect("create accuracy results");
    f.write_all(json.as_bytes()).expect("write accuracy results");
    println!("Results written to {}", path);

    assert!(all_pass, "One or more accuracy tests FAILED — see output above");
}
