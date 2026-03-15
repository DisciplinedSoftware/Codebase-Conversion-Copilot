use crate::types::{BlasFloat, Diag, Side, Trans, Uplo};

/// General matrix-matrix multiply: C = alpha*op(A)*op(B) + beta*C
/// Column-major: A[col*lda+row]
/// op(A) is m×k, op(B) is k×n, C is m×n
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn gemm<T: BlasFloat>(
    transa: Trans,
    transb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // Scale C by beta
    for j in 0..n {
        for i in 0..m {
            c[j * ldc + i] *= beta;
        }
    }
    if alpha == T::zero() {
        return;
    }
    for j in 0..n {
        for l in 0..k {
            // b_val = op(B)[l, j]
            let b_val = match transb {
                Trans::No => b[j * ldb + l],
                _ => b[l * ldb + j],
            };
            let ab = alpha * b_val;
            for i in 0..m {
                // a_val = op(A)[i, l]
                let a_val = match transa {
                    Trans::No => a[l * lda + i],
                    _ => a[i * lda + l],
                };
                c[j * ldc + i] += ab * a_val;
            }
        }
    }
}

/// Symmetric matrix-matrix: C = alpha*A*B + beta*C or alpha*B*A + beta*C
/// A is symmetric n×n (side Left: A is m×m, side Right: A is n×n)
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn symm<T: BlasFloat>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // Scale C
    for j in 0..n {
        for i in 0..m {
            c[j * ldc + i] *= beta;
        }
    }
    if alpha == T::zero() {
        return;
    }
    match side {
        Side::Left => {
            // C = alpha*A*B + beta*C, A is m×m symmetric
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();
                    for k in 0..m {
                        // A(i,k) symmetric
                        let a_val = if uplo == Uplo::Upper {
                            if k >= i {
                                a[k * lda + i]
                            } else {
                                a[i * lda + k]
                            }
                        } else {
                            if k <= i {
                                a[k * lda + i]
                            } else {
                                a[i * lda + k]
                            }
                        };
                        sum += a_val * b[j * ldb + k];
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
        Side::Right => {
            // C = alpha*B*A + beta*C, A is n×n symmetric
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();
                    for k in 0..n {
                        // A(k,j) symmetric
                        let a_val = if uplo == Uplo::Upper {
                            if j >= k {
                                a[j * lda + k]
                            } else {
                                a[k * lda + j]
                            }
                        } else {
                            if j <= k {
                                a[j * lda + k]
                            } else {
                                a[k * lda + j]
                            }
                        };
                        sum += b[k * ldb + i] * a_val;
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
    }
}

/// Symmetric rank-k update: C = alpha*op(A)*op(A)^T + beta*C
/// Trans::No: A is n×k, C is n×n
/// Trans::Yes: A is k×n, C is n×n
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn syrk<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // Scale relevant triangle of C
    match uplo {
        Uplo::Upper => {
            for j in 0..n {
                for i in 0..=j {
                    c[j * ldc + i] *= beta;
                }
            }
        }
        Uplo::Lower => {
            for j in 0..n {
                for i in j..n {
                    c[j * ldc + i] *= beta;
                }
            }
        }
    }
    if alpha == T::zero() {
        return;
    }
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            // C(i,j) += alpha * sum_l A(i,l)*A(j,l) for i<=j
            // A(i,l) = a[l*lda+i]
            for j in 0..n {
                for l in 0..k {
                    let ajl = alpha * a[l * lda + j];
                    for i in 0..=j {
                        c[j * ldc + i] += a[l * lda + i] * ajl;
                    }
                }
            }
        }
        (Uplo::Upper, _) => {
            // Trans: A is k×n, op(A) is n×k: A^T[i,l] = A[l,i] = a[i*lda+l]
            for j in 0..n {
                for i in 0..=j {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += a[i * lda + l] * a[j * lda + l];
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
        (Uplo::Lower, Trans::No) => {
            for j in 0..n {
                for l in 0..k {
                    let ajl = alpha * a[l * lda + j];
                    for i in j..n {
                        c[j * ldc + i] += a[l * lda + i] * ajl;
                    }
                }
            }
        }
        (Uplo::Lower, _) => {
            for j in 0..n {
                for i in j..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += a[i * lda + l] * a[j * lda + l];
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
    }
}

/// Symmetric rank-2k update:
/// C = alpha*op(A)*op(B)^T + alpha*op(B)*op(A)^T + beta*C
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn syr2k<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    match uplo {
        Uplo::Upper => {
            for j in 0..n {
                for i in 0..=j {
                    c[j * ldc + i] *= beta;
                }
            }
        }
        Uplo::Lower => {
            for j in 0..n {
                for i in j..n {
                    c[j * ldc + i] *= beta;
                }
            }
        }
    }
    if alpha == T::zero() {
        return;
    }
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            // op(A)=A (n×k), op(B)=B (n×k)
            for j in 0..n {
                for l in 0..k {
                    let ajl = alpha * a[l * lda + j];
                    let bjl = alpha * b[l * ldb + j];
                    for i in 0..=j {
                        c[j * ldc + i] += a[l * lda + i] * bjl + b[l * ldb + i] * ajl;
                    }
                }
            }
        }
        (Uplo::Upper, _) => {
            // op(A)=A^T (n×k → k×n transposed), A is k×n
            for j in 0..n {
                for i in 0..=j {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += a[i * lda + l] * b[j * ldb + l] + b[i * ldb + l] * a[j * lda + l];
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
        (Uplo::Lower, Trans::No) => {
            for j in 0..n {
                for l in 0..k {
                    let ajl = alpha * a[l * lda + j];
                    let bjl = alpha * b[l * ldb + j];
                    for i in j..n {
                        c[j * ldc + i] += a[l * lda + i] * bjl + b[l * ldb + i] * ajl;
                    }
                }
            }
        }
        (Uplo::Lower, _) => {
            for j in 0..n {
                for i in j..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += a[i * lda + l] * b[j * ldb + l] + b[i * ldb + l] * a[j * lda + l];
                    }
                    c[j * ldc + i] += alpha * sum;
                }
            }
        }
    }
}

/// Triangular matrix-matrix multiply
/// Side::Left:  B = alpha*op(A)*B
/// Side::Right: B = alpha*B*op(A)
/// A is triangular
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn trmm<T: BlasFloat>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match side {
        Side::Left => {
            // B = alpha*op(A)*B, A is m×m triangular
            match (uplo, trans) {
                (Uplo::Upper, Trans::No) => {
                    // Process column j of B
                    for j in 0..n {
                        // x = op(A) * b_j, upper no-trans
                        // Process rows right to left (like trmv upper no-trans)
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in i..m {
                                // A(i,k) = a[k*lda+i] for i<=k (upper)
                                let a_val = if nounit || i != k {
                                    a[k * lda + i]
                                } else {
                                    T::one()
                                };
                                sum += a_val * b[j * ldb + k];
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Upper, _) => {
                    // A^T is lower triangular
                    for j in 0..n {
                        for i in (0..m).rev() {
                            let mut sum = T::zero();
                            for k in 0..=i {
                                let a_val = if nounit || k != i {
                                    a[i * lda + k]
                                } else {
                                    T::one()
                                };
                                sum += a_val * b[j * ldb + k];
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Lower, Trans::No) => {
                    for j in 0..n {
                        for i in (0..m).rev() {
                            let mut sum = T::zero();
                            for k in 0..=i {
                                // A(i,k) = a[k*lda+i] for i>=k (lower)
                                let a_val = if nounit || i != k {
                                    a[k * lda + i]
                                } else {
                                    T::one()
                                };
                                sum += a_val * b[j * ldb + k];
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Lower, _) => {
                    // A^T is upper triangular
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in i..m {
                                let a_val = if nounit || k != i {
                                    a[i * lda + k]
                                } else {
                                    T::one()
                                };
                                sum += a_val * b[j * ldb + k];
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
            }
        }
        Side::Right => {
            // B = alpha*B*op(A), A is n×n triangular
            match (uplo, trans) {
                (Uplo::Upper, Trans::No) => {
                    // B_row_i = B_row_i * op(A) = B_row_i * A_upper
                    for i in 0..m {
                        for j in (0..n).rev() {
                            let mut sum = T::zero();
                            for k in 0..=j {
                                // A(k,j) = a[j*lda+k] for k<=j (upper)
                                let a_val = if nounit || k != j {
                                    a[j * lda + k]
                                } else {
                                    T::one()
                                };
                                sum += b[k * ldb + i] * a_val;
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Upper, _) => {
                    // B*A^T, A upper → A^T lower
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = T::zero();
                            for k in j..n {
                                // A^T(k,j) = A(j,k) = a[k*lda+j] for j<=k (upper)
                                let a_val = if nounit || j != k {
                                    a[k * lda + j]
                                } else {
                                    T::one()
                                };
                                sum += b[k * ldb + i] * a_val;
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Lower, Trans::No) => {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = T::zero();
                            for k in j..n {
                                // A(k,j) = a[j*lda+k] for k>=j (lower)
                                let a_val = if nounit || k != j {
                                    a[j * lda + k]
                                } else {
                                    T::one()
                                };
                                sum += b[k * ldb + i] * a_val;
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
                (Uplo::Lower, _) => {
                    // B*A^T, A lower → A^T upper
                    for i in 0..m {
                        for j in (0..n).rev() {
                            let mut sum = T::zero();
                            for k in 0..=j {
                                // A^T(k,j) = A(j,k) = a[k*lda+j] for j>=k (lower)
                                let a_val = if nounit || j != k {
                                    a[k * lda + j]
                                } else {
                                    T::one()
                                };
                                sum += b[k * ldb + i] * a_val;
                            }
                            b[j * ldb + i] = alpha * sum;
                        }
                    }
                }
            }
        }
    }
}

/// Triangular solve: op(A)*X = alpha*B or X*op(A) = alpha*B
/// Overwrites B with X
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn trsm<T: BlasFloat>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;

    // Scale B by alpha
    for j in 0..n {
        for i in 0..m {
            b[j * ldb + i] *= alpha;
        }
    }

    match side {
        Side::Left => {
            // Solve op(A)*X = B, A is m×m triangular
            match (uplo, trans) {
                (Uplo::Upper, Trans::No) => {
                    // Back substitution over rows of X (each column j)
                    for j in 0..n {
                        for r in (0..m).rev() {
                            if nounit {
                                b[j * ldb + r] /= a[r * lda + r];
                            }
                            let xrj = b[j * ldb + r];
                            for i in 0..r {
                                b[j * ldb + i] -= a[r * lda + i] * xrj;
                            }
                        }
                    }
                }
                (Uplo::Upper, _) => {
                    // Forward substitution, A^T is lower
                    for j in 0..n {
                        for r in 0..m {
                            let mut sum = T::zero();
                            for k in 0..r {
                                sum += a[r * lda + k] * b[j * ldb + k];
                            }
                            b[j * ldb + r] -= sum;
                            if nounit {
                                b[j * ldb + r] /= a[r * lda + r];
                            }
                        }
                    }
                }
                (Uplo::Lower, Trans::No) => {
                    // Forward substitution
                    for j in 0..n {
                        for r in 0..m {
                            let mut sum = T::zero();
                            for k in 0..r {
                                // A(r,k) = a[k*lda+r] lower tri
                                sum += a[k * lda + r] * b[j * ldb + k];
                            }
                            b[j * ldb + r] -= sum;
                            if nounit {
                                b[j * ldb + r] /= a[r * lda + r];
                            }
                        }
                    }
                }
                (Uplo::Lower, _) => {
                    // Back substitution, A^T is upper
                    for j in 0..n {
                        for r in (0..m).rev() {
                            let mut sum = T::zero();
                            for k in (r + 1)..m {
                                // A^T(r,k) = A(k,r) = a[r*lda+k] lower tri
                                sum += a[r * lda + k] * b[j * ldb + k];
                            }
                            b[j * ldb + r] -= sum;
                            if nounit {
                                b[j * ldb + r] /= a[r * lda + r];
                            }
                        }
                    }
                }
            }
        }
        Side::Right => {
            // Solve X*op(A) = B, A is n×n triangular
            match (uplo, trans) {
                (Uplo::Upper, Trans::No) => {
                    // Forward substitution over columns of X
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in 0..j {
                                // A(k,j) = a[j*lda+k] upper tri, k<=j
                                sum += b[k * ldb + i] * a[j * lda + k];
                            }
                            b[j * ldb + i] -= sum;
                            if nounit {
                                b[j * ldb + i] /= a[j * lda + j];
                            }
                        }
                    }
                }
                (Uplo::Upper, _) => {
                    // Back sub over cols: X*A^T=B, A^T lower
                    for j in (0..n).rev() {
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in (j + 1)..n {
                                // A^T(k,j) = A(j,k) = a[k*lda+j] upper tri
                                sum += b[k * ldb + i] * a[k * lda + j];
                            }
                            b[j * ldb + i] -= sum;
                            if nounit {
                                b[j * ldb + i] /= a[j * lda + j];
                            }
                        }
                    }
                }
                (Uplo::Lower, Trans::No) => {
                    // Back sub
                    for j in (0..n).rev() {
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in (j + 1)..n {
                                // A(k,j) = a[j*lda+k] lower tri, k>=j
                                sum += b[k * ldb + i] * a[j * lda + k];
                            }
                            b[j * ldb + i] -= sum;
                            if nounit {
                                b[j * ldb + i] /= a[j * lda + j];
                            }
                        }
                    }
                }
                (Uplo::Lower, _) => {
                    // Forward sub: X*A^T=B, A^T upper
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = T::zero();
                            for k in 0..j {
                                // A^T(k,j) = A(j,k) = a[k*lda+j] lower tri
                                sum += b[k * ldb + i] * a[k * lda + j];
                            }
                            b[j * ldb + i] -= sum;
                            if nounit {
                                b[j * ldb + i] /= a[j * lda + j];
                            }
                        }
                    }
                }
            }
        }
    }
}
