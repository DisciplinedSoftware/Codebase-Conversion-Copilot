use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_blas::level1;
use rust_blas::level2;
use rust_blas::level3;
use rust_blas::types::{Diag, Side, Trans, Uplo};
use std::time::Duration;

fn make_vec_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 1.0) / (n as f64)).collect()
}
fn make_vec_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 + 1.0) / (n as f32)).collect()
}
fn make_mat_f64(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|k| {
            let i = (k % rows) + 1;
            let j = (k / rows) + 1;
            ((i * j) as f64).sin()
        })
        .collect()
}
fn make_mat_f32(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|k| {
            let i = (k % rows) + 1;
            let j = (k / rows) + 1;
            ((i * j) as f32).sin()
        })
        .collect()
}
/// Build a well-conditioned upper triangular matrix (large diagonal)
fn make_tri_f64(n: usize) -> Vec<f64> {
    let mut a = make_mat_f64(n, n);
    for i in 0..n {
        a[i * n + i] = n as f64 + i as f64 + 1.0;
    }
    a
}
fn make_tri_f32(n: usize) -> Vec<f32> {
    let mut a = make_mat_f32(n, n);
    for i in 0..n {
        a[i * n + i] = n as f32 + i as f32 + 1.0;
    }
    a
}

// ─── Level 1 ────────────────────────────────────────────────────────────────

fn bench_l1_dasum(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/dasum");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::asum(n, &x, 1))
        });
    }
    g.finish();
}

fn bench_l1_sasum(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/sasum");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::asum(n, &x, 1))
        });
    }
    g.finish();
}

fn bench_l1_daxpy(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/daxpy");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f64(n);
            b.iter(|| level1::axpy(n, 2.0f64, &x, 1, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l1_saxpy(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/saxpy");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f32(n);
            b.iter(|| level1::axpy(n, 2.0f32, &x, 1, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l1_ddot(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/ddot");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f64(n);
        let y = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::dot(n, &x, 1, &y, 1))
        });
    }
    g.finish();
}

fn bench_l1_sdot(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/sdot");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f32(n);
        let y = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::dot(n, &x, 1, &y, 1))
        });
    }
    g.finish();
}

fn bench_l1_dnrm2(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/dnrm2");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::nrm2(n, &x, 1))
        });
    }
    g.finish();
}

fn bench_l1_snrm2(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/snrm2");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| level1::nrm2(n, &x, 1))
        });
    }
    g.finish();
}

fn bench_l1_dscal(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/dscal");
    for &n in &[64usize, 512, 2048] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f64(n);
            b.iter(|| level1::scal(n, 2.0f64, &mut x, 1))
        });
    }
    g.finish();
}

fn bench_l1_sscal(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/sscal");
    for &n in &[64usize, 512, 2048] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f32(n);
            b.iter(|| level1::scal(n, 2.0f32, &mut x, 1))
        });
    }
    g.finish();
}

fn bench_l1_dswap(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/dswap");
    for &n in &[64usize, 512, 2048] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f64(n);
            let mut y = make_vec_f64(n);
            b.iter(|| level1::swap(n, &mut x, 1, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l1_sswap(c: &mut Criterion) {
    let mut g = c.benchmark_group("level1/sswap");
    for &n in &[64usize, 512, 2048] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f32(n);
            let mut y = make_vec_f32(n);
            b.iter(|| level1::swap(n, &mut x, 1, &mut y, 1))
        });
    }
    g.finish();
}

// ─── Level 2 ────────────────────────────────────────────────────────────────

fn bench_l2_dgemv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/dgemv");
    for &n in &[64usize, 512, 2048] {
        let a = make_mat_f64(n, n);
        let x = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f64(n);
            b.iter(|| level2::gemv(Trans::No, n, n, 1.0, &a, n, &x, 1, 0.0, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l2_sgemv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/sgemv");
    for &n in &[64usize, 512, 2048] {
        let a = make_mat_f32(n, n);
        let x = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f32(n);
            b.iter(|| level2::gemv(Trans::No, n, n, 1.0f32, &a, n, &x, 1, 0.0f32, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l2_dger(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/dger");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f64(n);
        let y = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut a = make_mat_f64(n, n);
            b.iter(|| level2::ger(n, n, 1.0, &x, 1, &y, 1, &mut a, n))
        });
    }
    g.finish();
}

fn bench_l2_sger(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/sger");
    for &n in &[64usize, 512, 2048] {
        let x = make_vec_f32(n);
        let y = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut a = make_mat_f32(n, n);
            b.iter(|| level2::ger(n, n, 1.0f32, &x, 1, &y, 1, &mut a, n))
        });
    }
    g.finish();
}

fn bench_l2_dsymv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/dsymv");
    for &n in &[64usize, 512, 2048] {
        let a = make_mat_f64(n, n);
        let x = make_vec_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f64(n);
            b.iter(|| level2::symv(Uplo::Upper, n, 1.0, &a, n, &x, 1, 0.0, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l2_ssymv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/ssymv");
    for &n in &[64usize, 512, 2048] {
        let a = make_mat_f32(n, n);
        let x = make_vec_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut y = make_vec_f32(n);
            b.iter(|| level2::symv(Uplo::Upper, n, 1.0f32, &a, n, &x, 1, 0.0f32, &mut y, 1))
        });
    }
    g.finish();
}

fn bench_l2_dtrmv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/dtrmv");
    for &n in &[64usize, 512, 2048] {
        let a = make_tri_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f64(n);
            b.iter(|| level2::trmv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut x, 1))
        });
    }
    g.finish();
}

fn bench_l2_strmv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/strmv");
    for &n in &[64usize, 512, 2048] {
        let a = make_tri_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f32(n);
            b.iter(|| {
                level2::trmv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut x, 1)
            })
        });
    }
    g.finish();
}

fn bench_l2_dtrsv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/dtrsv");
    for &n in &[64usize, 512, 2048] {
        let a = make_tri_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f64(n);
            b.iter(|| level2::trsv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut x, 1))
        });
    }
    g.finish();
}

fn bench_l2_strsv(c: &mut Criterion) {
    let mut g = c.benchmark_group("level2/strsv");
    for &n in &[64usize, 512, 2048] {
        let a = make_tri_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut x = make_vec_f32(n);
            b.iter(|| {
                level2::trsv(Uplo::Upper, Trans::No, Diag::NonUnit, n, &a, n, &mut x, 1)
            })
        });
    }
    g.finish();
}

// ─── Level 3 ────────────────────────────────────────────────────────────────

fn bench_l3_dgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/dgemm");
    for &n in &[64usize, 128, 256] {
        let a = make_mat_f64(n, n);
        let b = make_mat_f64(n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut cc = make_mat_f64(n, n);
            bench.iter(|| {
                level3::gemm(Trans::No, Trans::No, n, n, n, 1.0, &a, n, &b, n, 0.0, &mut cc, n)
            })
        });
    }
    g.finish();
}

fn bench_l3_sgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/sgemm");
    for &n in &[64usize, 128, 256] {
        let a = make_mat_f32(n, n);
        let b = make_mat_f32(n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut cc = make_mat_f32(n, n);
            bench.iter(|| {
                level3::gemm(
                    Trans::No, Trans::No, n, n, n, 1.0f32, &a, n, &b, n, 0.0f32, &mut cc, n,
                )
            })
        });
    }
    g.finish();
}

fn bench_l3_dsyrk(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/dsyrk");
    for &n in &[64usize, 128, 256] {
        let a = make_mat_f64(n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut cc = make_mat_f64(n, n);
            bench.iter(|| {
                level3::syrk(Uplo::Upper, Trans::No, n, n, 1.0, &a, n, 0.0, &mut cc, n)
            })
        });
    }
    g.finish();
}

fn bench_l3_ssyrk(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/ssyrk");
    for &n in &[64usize, 128, 256] {
        let a = make_mat_f32(n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut cc = make_mat_f32(n, n);
            bench.iter(|| {
                level3::syrk(Uplo::Upper, Trans::No, n, n, 1.0f32, &a, n, 0.0f32, &mut cc, n)
            })
        });
    }
    g.finish();
}

fn bench_l3_dtrmm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/dtrmm");
    for &n in &[64usize, 128, 256] {
        let a = make_tri_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut b = make_mat_f64(n, n);
            bench.iter(|| {
                level3::trmm(
                    Side::Left,
                    Uplo::Upper,
                    Trans::No,
                    Diag::NonUnit,
                    n,
                    n,
                    1.0,
                    &a,
                    n,
                    &mut b,
                    n,
                )
            })
        });
    }
    g.finish();
}

fn bench_l3_strmm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/strmm");
    for &n in &[64usize, 128, 256] {
        let a = make_tri_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut b = make_mat_f32(n, n);
            bench.iter(|| {
                level3::trmm(
                    Side::Left,
                    Uplo::Upper,
                    Trans::No,
                    Diag::NonUnit,
                    n,
                    n,
                    1.0f32,
                    &a,
                    n,
                    &mut b,
                    n,
                )
            })
        });
    }
    g.finish();
}

fn bench_l3_dtrsm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/dtrsm");
    for &n in &[64usize, 128, 256] {
        let a = make_tri_f64(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut b = make_mat_f64(n, n);
            bench.iter(|| {
                level3::trsm(
                    Side::Left,
                    Uplo::Upper,
                    Trans::No,
                    Diag::NonUnit,
                    n,
                    n,
                    1.0,
                    &a,
                    n,
                    &mut b,
                    n,
                )
            })
        });
    }
    g.finish();
}

fn bench_l3_strsm(c: &mut Criterion) {
    let mut g = c.benchmark_group("level3/strsm");
    for &n in &[64usize, 128, 256] {
        let a = make_tri_f32(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            let mut b = make_mat_f32(n, n);
            bench.iter(|| {
                level3::trsm(
                    Side::Left,
                    Uplo::Upper,
                    Trans::No,
                    Diag::NonUnit,
                    n,
                    n,
                    1.0f32,
                    &a,
                    n,
                    &mut b,
                    n,
                )
            })
        });
    }
    g.finish();
}

criterion_group!(
    name = level1_benches;
    config = Criterion::default().measurement_time(Duration::from_secs(3)).warm_up_time(Duration::from_secs(1));
    targets = bench_l1_dasum, bench_l1_sasum, bench_l1_daxpy, bench_l1_saxpy,
              bench_l1_ddot, bench_l1_sdot, bench_l1_dnrm2, bench_l1_snrm2,
              bench_l1_dscal, bench_l1_sscal, bench_l1_dswap, bench_l1_sswap
);

criterion_group!(
    name = level2_benches;
    config = Criterion::default().measurement_time(Duration::from_secs(3)).warm_up_time(Duration::from_secs(1));
    targets = bench_l2_dgemv, bench_l2_sgemv, bench_l2_dger, bench_l2_sger,
              bench_l2_dsymv, bench_l2_ssymv, bench_l2_dtrmv, bench_l2_strmv,
              bench_l2_dtrsv, bench_l2_strsv
);

criterion_group!(
    name = level3_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1))
        .sample_size(50);
    targets = bench_l3_dgemm, bench_l3_sgemm, bench_l3_dsyrk, bench_l3_ssyrk,
              bench_l3_dtrmm, bench_l3_strmm, bench_l3_dtrsm, bench_l3_strsm
);

criterion_main!(level1_benches, level2_benches, level3_benches);
