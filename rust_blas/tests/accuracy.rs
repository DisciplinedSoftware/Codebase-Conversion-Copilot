/// Accuracy tests: compare Rust BLAS implementations against the reference
/// Fortran BLAS (linked via build.rs). Results written to /tmp/accuracy_results.json.
use rust_blas::level1;
use rust_blas::level2;
use rust_blas::level3;
use rust_blas::types::{Diag, Side, Trans, Uplo};
use std::io::Write;

// ── Fortran BLAS FFI (names end with underscore) ──────────────────────────
extern "C" {
    fn dasum_(n: *const i32, dx: *const f64, incx: *const i32) -> f64;
    fn daxpy_(
        n: *const i32,
        da: *const f64,
        dx: *const f64,
        incx: *const i32,
        dy: *mut f64,
        incy: *const i32,
    );
    fn ddot_(
        n: *const i32,
        dx: *const f64,
        incx: *const i32,
        dy: *const f64,
        incy: *const i32,
    ) -> f64;
    fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn dscal_(n: *const i32, da: *const f64, dx: *mut f64, incx: *const i32);
    fn dswap_(
        n: *const i32,
        dx: *mut f64,
        incx: *const i32,
        dy: *mut f64,
        incy: *const i32,
    );
    fn dgemv_(
        trans: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn dger_(
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
    fn dsymv_(
        uplo: *const u8,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn dtrmv_(
        uplo: *const u8,
        trans: *const u8,
        diag: *const u8,
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn dtrsv_(
        uplo: *const u8,
        trans: *const u8,
        diag: *const u8,
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn dgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsymm_(
        side: *const u8,
        uplo: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsyrk_(
        uplo: *const u8,
        trans: *const u8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dtrmm_(
        side: *const u8,
        uplo: *const u8,
        transa: *const u8,
        diag: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
    );
    fn dtrsm_(
        side: *const u8,
        uplo: *const u8,
        transa: *const u8,
        diag: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_vec(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64) / (n as f64)).collect()
}

fn make_mat(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|k| {
            let i = (k % rows) + 1;
            let j = (k / rows) + 1;
            ((i * j) as f64 * 0.1).sin()
        })
        .collect()
}

/// Well-conditioned upper triangular matrix (strong diagonal)
fn make_tri(n: usize) -> Vec<f64> {
    let mut a = make_mat(n, n);
    // Zero the lower triangle so the matrix is strictly upper triangular with large diagonal
    for j in 0..n {
        for i in (j + 1)..n {
            a[j * n + i] = 0.0;
        }
        a[j * n + j] = n as f64 + j as f64 + 1.0;
    }
    a
}

fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let denom = x.abs().max(y.abs()).max(1e-300);
            (x - y).abs() / denom
        })
        .fold(0.0_f64, f64::max)
}

const EPS: f64 = 1e-10;
const N: usize = 64;

struct AccResult {
    routine: String,
    precision: String,
    n: usize,
    max_rel_error: f64,
    passed: bool,
}

// ── Individual tests ──────────────────────────────────────────────────────

fn test_asum() -> AccResult {
    let x = make_vec(N);
    let ref_val = unsafe { dasum_(&(N as i32), x.as_ptr(), &1) };
    let rust_val = level1::asum(N, &x, 1);
    let err = (ref_val - rust_val).abs() / ref_val.abs().max(1e-300);
    AccResult {
        routine: "asum".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_axpy() -> AccResult {
    let x = make_vec(N);
    let y0 = make_vec(N);
    let alpha = 2.5;

    let mut y_ref = y0.clone();
    unsafe { daxpy_(&(N as i32), &alpha, x.as_ptr(), &1, y_ref.as_mut_ptr(), &1) };

    let mut y_rust = y0.clone();
    level1::axpy(N, alpha, &x, 1, &mut y_rust, 1);

    let err = max_rel_err(&y_ref, &y_rust);
    AccResult {
        routine: "axpy".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_dot() -> AccResult {
    let x = make_vec(N);
    let y = make_vec(N);
    let ref_val = unsafe { ddot_(&(N as i32), x.as_ptr(), &1, y.as_ptr(), &1) };
    let rust_val = level1::dot(N, &x, 1, &y, 1);
    let err = (ref_val - rust_val).abs() / ref_val.abs().max(1e-300);
    AccResult {
        routine: "dot".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_nrm2() -> AccResult {
    let x = make_vec(N);
    let ref_val = unsafe { dnrm2_(&(N as i32), x.as_ptr(), &1) };
    let rust_val = level1::nrm2(N, &x, 1);
    let err = (ref_val - rust_val).abs() / ref_val.abs().max(1e-300);
    AccResult {
        routine: "nrm2".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_scal() -> AccResult {
    let x0 = make_vec(N);
    let alpha = 3.14;

    let mut x_ref = x0.clone();
    unsafe { dscal_(&(N as i32), &alpha, x_ref.as_mut_ptr(), &1) };

    let mut x_rust = x0.clone();
    level1::scal(N, alpha, &mut x_rust, 1);

    let err = max_rel_err(&x_ref, &x_rust);
    AccResult {
        routine: "scal".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_swap() -> AccResult {
    let x0 = make_vec(N);
    let y0: Vec<f64> = (0..N).map(|i| (i as f64) * 0.3).collect();

    let mut x_ref = x0.clone();
    let mut y_ref = y0.clone();
    unsafe { dswap_(&(N as i32), x_ref.as_mut_ptr(), &1, y_ref.as_mut_ptr(), &1) };

    let mut x_rust = x0.clone();
    let mut y_rust = y0.clone();
    level1::swap(N, &mut x_rust, 1, &mut y_rust, 1);

    let err = max_rel_err(&x_ref, &x_rust).max(max_rel_err(&y_ref, &y_rust));
    AccResult {
        routine: "swap".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_gemv() -> AccResult {
    let a = make_mat(N, N);
    let x = make_vec(N);
    let y0 = make_vec(N);
    let alpha = 1.5;
    let beta = 0.5;

    let mut y_ref = y0.clone();
    unsafe {
        dgemv_(
            b"N".as_ptr(),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            x.as_ptr(),
            &1,
            &beta,
            y_ref.as_mut_ptr(),
            &1,
        )
    };

    let mut y_rust = y0.clone();
    level2::gemv(Trans::No, N, N, alpha, &a, N, &x, 1, beta, &mut y_rust, 1);

    let err = max_rel_err(&y_ref, &y_rust);
    AccResult {
        routine: "gemv".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_ger() -> AccResult {
    let x = make_vec(N);
    let y = make_vec(N);
    let a0 = make_mat(N, N);
    let alpha = 0.7;

    let mut a_ref = a0.clone();
    unsafe {
        dger_(
            &(N as i32),
            &(N as i32),
            &alpha,
            x.as_ptr(),
            &1,
            y.as_ptr(),
            &1,
            a_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut a_rust = a0.clone();
    level2::ger(N, N, alpha, &x, 1, &y, 1, &mut a_rust, N);

    let err = max_rel_err(&a_ref, &a_rust);
    AccResult {
        routine: "ger".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_symv() -> AccResult {
    let a = make_mat(N, N);
    let x = make_vec(N);
    let y0 = make_vec(N);
    let alpha = 1.2;
    let beta = 0.3;

    let mut y_ref = y0.clone();
    unsafe {
        dsymv_(
            b"U".as_ptr(),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            x.as_ptr(),
            &1,
            &beta,
            y_ref.as_mut_ptr(),
            &1,
        )
    };

    let mut y_rust = y0.clone();
    level2::symv(Uplo::Upper, N, alpha, &a, N, &x, 1, beta, &mut y_rust, 1);

    let err = max_rel_err(&y_ref, &y_rust);
    AccResult {
        routine: "symv".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_trmv() -> AccResult {
    let a = make_tri(N);
    let x0 = make_vec(N);

    let mut x_ref = x0.clone();
    unsafe {
        dtrmv_(
            b"U".as_ptr(),
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            a.as_ptr(),
            &(N as i32),
            x_ref.as_mut_ptr(),
            &1,
        )
    };

    let mut x_rust = x0.clone();
    level2::trmv(Uplo::Upper, Trans::No, Diag::NonUnit, N, &a, N, &mut x_rust, 1);

    let err = max_rel_err(&x_ref, &x_rust);
    AccResult {
        routine: "trmv".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_trsv() -> AccResult {
    let a = make_tri(N);
    let x0 = make_vec(N);

    let mut x_ref = x0.clone();
    unsafe {
        dtrsv_(
            b"U".as_ptr(),
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            a.as_ptr(),
            &(N as i32),
            x_ref.as_mut_ptr(),
            &1,
        )
    };

    let mut x_rust = x0.clone();
    level2::trsv(Uplo::Upper, Trans::No, Diag::NonUnit, N, &a, N, &mut x_rust, 1);

    let err = max_rel_err(&x_ref, &x_rust);
    AccResult {
        routine: "trsv".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_gemm() -> AccResult {
    let a = make_mat(N, N);
    let b = make_mat(N, N);
    let c0 = make_mat(N, N);
    let alpha = 1.0;
    let beta = 0.0;

    let mut c_ref = c0.clone();
    unsafe {
        dgemm_(
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            b.as_ptr(),
            &(N as i32),
            &beta,
            c_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut c_rust = c0.clone();
    level3::gemm(Trans::No, Trans::No, N, N, N, alpha, &a, N, &b, N, beta, &mut c_rust, N);

    let err = max_rel_err(&c_ref, &c_rust);
    AccResult {
        routine: "gemm".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_symm() -> AccResult {
    let a = make_mat(N, N);
    let b = make_mat(N, N);
    let c0 = make_mat(N, N);
    let alpha = 1.0;
    let beta = 0.0;

    let mut c_ref = c0.clone();
    unsafe {
        dsymm_(
            b"L".as_ptr(),
            b"U".as_ptr(),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            b.as_ptr(),
            &(N as i32),
            &beta,
            c_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut c_rust = c0.clone();
    level3::symm(Side::Left, Uplo::Upper, N, N, alpha, &a, N, &b, N, beta, &mut c_rust, N);

    let err = max_rel_err(&c_ref, &c_rust);
    AccResult {
        routine: "symm".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_syrk() -> AccResult {
    let a = make_mat(N, N);
    let c0 = make_mat(N, N);
    let alpha = 1.0;
    let beta = 0.0;

    let mut c_ref = c0.clone();
    unsafe {
        dsyrk_(
            b"U".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            &beta,
            c_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut c_rust = c0.clone();
    level3::syrk(Uplo::Upper, Trans::No, N, N, alpha, &a, N, beta, &mut c_rust, N);

    // Only compare upper triangle
    let mut max_err = 0.0f64;
    for j in 0..N {
        for i in 0..=j {
            let r = &c_ref[j * N + i];
            let t = &c_rust[j * N + i];
            let e = (r - t).abs() / r.abs().max(t.abs()).max(1e-300);
            if e > max_err {
                max_err = e;
            }
        }
    }
    AccResult {
        routine: "syrk".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: max_err,
        passed: max_err < EPS,
    }
}

fn test_trmm() -> AccResult {
    let a = make_tri(N);
    let b0 = make_mat(N, N);
    let alpha = 1.0;

    let mut b_ref = b0.clone();
    unsafe {
        dtrmm_(
            b"L".as_ptr(),
            b"U".as_ptr(),
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            b_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut b_rust = b0.clone();
    level3::trmm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, N, N, alpha, &a, N, &mut b_rust, N);

    let err = max_rel_err(&b_ref, &b_rust);
    AccResult {
        routine: "trmm".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

fn test_trsm() -> AccResult {
    let a = make_tri(N);
    let b0 = make_mat(N, N);
    let alpha = 1.0;

    let mut b_ref = b0.clone();
    unsafe {
        dtrsm_(
            b"L".as_ptr(),
            b"U".as_ptr(),
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(N as i32),
            &(N as i32),
            &alpha,
            a.as_ptr(),
            &(N as i32),
            b_ref.as_mut_ptr(),
            &(N as i32),
        )
    };

    let mut b_rust = b0.clone();
    level3::trsm(Side::Left, Uplo::Upper, Trans::No, Diag::NonUnit, N, N, alpha, &a, N, &mut b_rust, N);

    let err = max_rel_err(&b_ref, &b_rust);
    AccResult {
        routine: "trsm".into(),
        precision: "f64".into(),
        n: N,
        max_rel_error: err,
        passed: err < EPS,
    }
}

// ── Main test that writes JSON ────────────────────────────────────────────

#[test]
fn write_accuracy_report() {
    let results = vec![
        test_asum(),
        test_axpy(),
        test_dot(),
        test_nrm2(),
        test_scal(),
        test_swap(),
        test_gemv(),
        test_ger(),
        test_symv(),
        test_trmv(),
        test_trsv(),
        test_gemm(),
        test_symm(),
        test_syrk(),
        test_trmm(),
        test_trsm(),
    ];

    // Print summary
    println!("\n=== Accuracy Report ===");
    let mut all_pass = true;
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "{:8} {:4} n={:4}  max_rel_err = {:.3e}  [{}]",
            r.routine, r.precision, r.n, r.max_rel_error, status
        );
        if !r.passed {
            all_pass = false;
        }
    }

    // Write JSON
    let mut json = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        json.push_str(&format!(
            "  {{\"routine\": \"{}\", \"precision\": \"{}\", \"n\": {}, \"max_rel_error\": {:.6e}, \"passed\": {}}}{}\n",
            r.routine, r.precision, r.n, r.max_rel_error, r.passed, comma
        ));
    }
    json.push_str("]\n");

    let path = "/tmp/accuracy_results.json";
    let mut f = std::fs::File::create(path).expect("create accuracy results");
    f.write_all(json.as_bytes()).expect("write accuracy results");
    println!("Results written to {}", path);

    assert!(all_pass, "Some accuracy tests failed — see output above");
}
