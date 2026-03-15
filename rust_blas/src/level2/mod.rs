use crate::types::{BlasFloat, Diag, Trans, Uplo};

/// General matrix-vector multiply: y = alpha*op(A)*x + beta*y
/// Column-major storage: A[col * lda + row]
pub fn gemv<T: BlasFloat>(
    trans: Trans,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let leny = if trans == Trans::No { m } else { n };
    let mut iy = 0;
    for _ in 0..leny {
        y[iy] = beta * y[iy];
        iy += incy;
    }
    if alpha == T::zero() {
        return;
    }
    match trans {
        Trans::No => {
            // y(i) += alpha * sum_j A(i,j)*x(j), A(i,j) = a[j*lda+i]
            let mut jx = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let mut iy2 = 0;
                for i in 0..m {
                    y[iy2] += xj * a[j * lda + i];
                    iy2 += incy;
                }
                jx += incx;
            }
        }
        _ => {
            // y(j) += alpha * sum_i A(i,j)*x(i), A(i,j) = a[j*lda+i]
            let mut jy = 0;
            for j in 0..n {
                let mut sum = T::zero();
                let mut ix2 = 0;
                for i in 0..m {
                    sum += a[j * lda + i] * x[ix2];
                    ix2 += incx;
                }
                y[jy] += alpha * sum;
                jy += incy;
            }
        }
    }
}

/// Banded matrix-vector multiply
/// Band storage: a[j*lda + (ku+i-j)] for element (row i, col j), valid when max(0,j-ku)<=i<=min(m-1,j+kl)
pub fn gbmv<T: BlasFloat>(
    trans: Trans,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let leny = if trans == Trans::No { m } else { n };
    let mut iy = 0;
    for _ in 0..leny {
        y[iy] = beta * y[iy];
        iy += incy;
    }
    if alpha == T::zero() {
        return;
    }
    match trans {
        Trans::No => {
            let mut jx = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let i_min = if j >= ku { j - ku } else { 0 };
                let i_max = (j + kl).min(m - 1);
                let mut iy2 = i_min * incy;
                for i in i_min..=i_max {
                    y[iy2] += xj * a[j * lda + ku + i - j];
                    iy2 += incy;
                }
                jx += incx;
            }
        }
        _ => {
            let mut ix = 0;
            for i in 0..m {
                let xi = alpha * x[ix];
                let j_min = if i >= kl { i - kl } else { 0 };
                let j_max = (i + ku).min(n - 1);
                let mut jy = j_min * incy;
                for j in j_min..=j_max {
                    y[jy] += xi * a[j * lda + ku + i - j];
                    jy += incy;
                }
                ix += incx;
            }
        }
    }
}

/// Symmetric matrix-vector: y = alpha*A*x + beta*y, A symmetric n×n
pub fn symv<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = beta * y[iy];
        iy += incy;
    }
    if alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            // A(i,j) for i<=j stored at a[j*lda+i]
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let mut sum = T::zero();
                let mut ix2 = 0;
                let mut iy2 = 0;
                for i in 0..j {
                    // off-diagonal: symmetric so A(i,j) = A(j,i)
                    y[iy2] += alpha * a[j * lda + i] * x[jx];
                    sum += a[j * lda + i] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                // diagonal
                sum += a[j * lda + j] * x[jx];
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
        Uplo::Lower => {
            // A(i,j) for i>=j stored at a[j*lda+i]
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let mut sum = a[j * lda + j] * x[jx];
                let mut ix2 = (j + 1) * incx;
                let mut iy2 = (j + 1) * incy;
                for i in (j + 1)..n {
                    y[iy2] += alpha * a[j * lda + i] * x[jx];
                    sum += a[j * lda + i] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
    }
}

/// Symmetric banded matrix-vector
/// Upper band storage: a[j*lda + (k+i-j)] for i<=j, i>=j-k (k superdiagonals)
/// Lower band storage: a[j*lda + (i-j)] for i>=j, i<=j+k
pub fn sbmv<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = beta * y[iy];
        iy += incy;
    }
    if alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                // diagonal stored at row index k
                let mut sum = a[j * lda + k] * x[jx];
                let i_min = if j >= k { j - k } else { 0 };
                let mut ix2 = i_min * incx;
                let mut iy2 = i_min * incy;
                for i in i_min..j {
                    let band_row = k + i - j;
                    y[iy2] += alpha * a[j * lda + band_row] * x[jx];
                    sum += a[j * lda + band_row] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                // diagonal at row index 0
                let mut sum = a[j * lda] * x[jx];
                let i_max = (j + k).min(n - 1);
                let mut ix2 = (j + 1) * incx;
                let mut iy2 = (j + 1) * incy;
                for i in (j + 1)..=i_max {
                    let band_row = i - j;
                    y[iy2] += alpha * a[j * lda + band_row] * x[jx];
                    sum += a[j * lda + band_row] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
    }
}

/// Symmetric packed matrix-vector
/// Upper: ap[j*(j+1)/2 + i] for i<=j (0-based)
/// Lower: ap[j*(2*n-j-1)/2 + (i-j)] = ap[j*(2*n-j-1)/2 + i - j] for i>=j
pub fn spmv<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = beta * y[iy];
        iy += incy;
    }
    if alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let mut sum = T::zero();
                let mut ix2 = 0;
                let mut iy2 = 0;
                for i in 0..j {
                    let idx = j * (j + 1) / 2 + i;
                    y[iy2] += alpha * ap[idx] * x[jx];
                    sum += ap[idx] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                let diag_idx = j * (j + 1) / 2 + j;
                sum += ap[diag_idx] * x[jx];
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                // col_start = j*(2*n-j-1)/2, element (i,j) at col_start + (i-j)
                // But we need to be careful: j*(2*n-j-1) must be even
                // For j=0: 0*(2n-1)/2 = 0, elements: i=0..n-1
                // Actually use: col_start index for col j = sum_{c=0}^{j-1}(n-c) = j*n - j*(j-1)/2
                let col_start = j * n - j * (j - 1) / 2;
                // diagonal at offset 0 from col_start
                let mut sum = ap[col_start] * x[jx];
                let mut ix2 = (j + 1) * incx;
                let mut iy2 = (j + 1) * incy;
                for i in (j + 1)..n {
                    let idx = col_start + (i - j);
                    y[iy2] += alpha * ap[idx] * x[jx];
                    sum += ap[idx] * x[ix2];
                    ix2 += incx;
                    iy2 += incy;
                }
                y[jy] += alpha * sum;
                jx += incx;
                jy += incy;
            }
        }
    }
}

/// Triangular matrix-vector: x = op(A)*x
/// Column-major: A(i,j) = a[j*lda+i]
pub fn trmv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            // x = A*x, upper tri, column by column left to right
            let mut jx = 0;
            for j in 0..n {
                let temp = x[jx];
                let mut ix = 0;
                for i in 0..j {
                    x[ix] += a[j * lda + i] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = a[j * lda + j] * temp;
                }
                jx += incx;
            }
        }
        (Uplo::Upper, _) => {
            // x = A^T*x, upper tri, process j right to left
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let mut temp = if nounit { a[j * lda + j] * x[jx] } else { x[jx] };
                let mut ix = 0;
                for i in 0..j {
                    temp += a[j * lda + i] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, Trans::No) => {
            // x = A*x, lower tri, column by column right to left
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let temp = x[jx];
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..n {
                    x[ix] += a[j * lda + i] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = a[j * lda + j] * temp;
                }
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, _) => {
            // x = A^T*x, lower tri, process j left to right
            let mut jx = 0;
            for j in 0..n {
                let mut temp = if nounit { a[j * lda + j] * x[jx] } else { x[jx] };
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..n {
                    temp += a[j * lda + i] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                jx += incx;
            }
        }
    }
}

/// Triangular banded matrix-vector: x = op(A)*x
/// Upper band: a[j*lda + (k+i-j)] for i in max(0,j-k)..=j
/// Lower band: a[j*lda + (i-j)] for i in j..=min(n-1,j+k)
pub fn tbmv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            let mut jx = 0;
            for j in 0..n {
                let temp = x[jx];
                let i_min = if j >= k { j - k } else { 0 };
                let mut ix = i_min * incx;
                for i in i_min..j {
                    x[ix] += a[j * lda + k + i - j] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = a[j * lda + k] * temp;
                }
                jx += incx;
            }
        }
        (Uplo::Upper, _) => {
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let mut temp = if nounit { a[j * lda + k] * x[jx] } else { x[jx] };
                let i_min = if j >= k { j - k } else { 0 };
                let mut ix = i_min * incx;
                for i in i_min..j {
                    temp += a[j * lda + k + i - j] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, Trans::No) => {
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let temp = x[jx];
                let i_max = (j + k).min(n - 1);
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..=i_max {
                    x[ix] += a[j * lda + (i - j)] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = a[j * lda] * temp;
                }
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, _) => {
            let mut jx = 0;
            for j in 0..n {
                let mut temp = if nounit { a[j * lda] * x[jx] } else { x[jx] };
                let i_max = (j + k).min(n - 1);
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..=i_max {
                    temp += a[j * lda + (i - j)] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                jx += incx;
            }
        }
    }
}

/// Triangular packed matrix-vector: x = op(A)*x
/// Upper: ap[j*(j+1)/2 + i] for i<=j
/// Lower: ap[col_start_j + (i-j)] where col_start_j = j*n - j*(j-1)/2
pub fn tpmv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            let mut jx = 0;
            for j in 0..n {
                let temp = x[jx];
                let mut ix = 0;
                for i in 0..j {
                    x[ix] += ap[j * (j + 1) / 2 + i] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = ap[j * (j + 1) / 2 + j] * temp;
                }
                jx += incx;
            }
        }
        (Uplo::Upper, _) => {
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let diag_val = ap[j * (j + 1) / 2 + j];
                let mut temp = if nounit { diag_val * x[jx] } else { x[jx] };
                let mut ix = 0;
                for i in 0..j {
                    temp += ap[j * (j + 1) / 2 + i] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, Trans::No) => {
            let mut jx = (n - 1) * incx;
            for j in (0..n).rev() {
                let col_start = j * n - j * (j - 1) / 2;
                let temp = x[jx];
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..n {
                    x[ix] += ap[col_start + (i - j)] * temp;
                    ix += incx;
                }
                if nounit {
                    x[jx] = ap[col_start] * temp;
                }
                if j > 0 {
                    jx -= incx;
                }
            }
        }
        (Uplo::Lower, _) => {
            let mut jx = 0;
            for j in 0..n {
                let col_start = j * n - j * (j - 1) / 2;
                let mut temp = if nounit { ap[col_start] * x[jx] } else { x[jx] };
                let mut ix = (j + 1) * incx;
                for i in (j + 1)..n {
                    temp += ap[col_start + (i - j)] * x[ix];
                    ix += incx;
                }
                x[jx] = temp;
                jx += incx;
            }
        }
    }
}

/// Triangular solve: x = op(A)^{-1} * x
/// Column-major: A(i,j) = a[j*lda+i]
pub fn trsv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            // Back substitution: x[n-1..0]
            // x[r] = (x[r] - sum_{c=r+1}^{n-1} A(r,c)*x[c]) / A(r,r)
            // A(r,c) = a[c*lda+r] for r<=c (upper tri)
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let mut sum = T::zero();
                let mut cx = (r + 1) * incx;
                for c in (r + 1)..n {
                    sum += a[c * lda + r] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + r];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
        (Uplo::Upper, _) => {
            // Forward substitution: solve A^T*x = b
            // x[r] = (x[r] - sum_{c=0}^{r-1} A(c,r)*x[c]) / A(r,r)
            // A(c,r) = a[r*lda+c] (upper tri, c<=r)
            let mut rx = 0;
            for r in 0..n {
                let mut sum = T::zero();
                let mut cx = 0;
                for c in 0..r {
                    sum += a[r * lda + c] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + r];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, Trans::No) => {
            // Forward substitution: x[0..n-1]
            // x[r] = (x[r] - sum_{c=0}^{r-1} A(r,c)*x[c]) / A(r,r)
            // A(r,c) = a[c*lda+r] for c<=r (lower tri)
            let mut rx = 0;
            for r in 0..n {
                let mut sum = T::zero();
                let mut cx = 0;
                for c in 0..r {
                    sum += a[c * lda + r] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + r];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, _) => {
            // Back substitution: solve A^T*x = b
            // x[r] = (x[r] - sum_{c=r+1}^{n-1} A(c,r)*x[c]) / A(r,r)
            // A(c,r) = a[r*lda+c] (lower tri, c>=r)
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let mut sum = T::zero();
                let mut cx = (r + 1) * incx;
                for c in (r + 1)..n {
                    sum += a[r * lda + c] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + r];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
    }
}

/// Triangular banded solve: x = op(A)^{-1} * x
pub fn tbsv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let c_min = r + 1;
                let c_max = (r + k).min(n - 1);
                let mut sum = T::zero();
                let mut cx = c_min * incx;
                for c in c_min..=c_max {
                    sum += a[c * lda + k + r - c] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + k];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
        (Uplo::Upper, _) => {
            let mut rx = 0;
            for r in 0..n {
                let c_min = if r >= k { r - k } else { 0 };
                let mut sum = T::zero();
                let mut cx = c_min * incx;
                for c in c_min..r {
                    sum += a[r * lda + k + c - r] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda + k];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, Trans::No) => {
            let mut rx = 0;
            for r in 0..n {
                let c_min = if r >= k { r - k } else { 0 };
                let mut sum = T::zero();
                let mut cx = c_min * incx;
                for c in c_min..r {
                    sum += a[c * lda + (r - c)] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, _) => {
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let c_max = (r + k).min(n - 1);
                let mut sum = T::zero();
                let mut cx = (r + 1) * incx;
                for c in (r + 1)..=c_max {
                    sum += a[r * lda + (c - r)] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / a[r * lda];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
    }
}

/// Triangular packed solve: x = op(A)^{-1} * x
pub fn tpsv<T: BlasFloat>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) {
    if n == 0 {
        return;
    }
    let nounit = diag == Diag::NonUnit;
    match (uplo, trans) {
        (Uplo::Upper, Trans::No) => {
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let mut sum = T::zero();
                let mut cx = (r + 1) * incx;
                for c in (r + 1)..n {
                    sum += ap[c * (c + 1) / 2 + r] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / ap[r * (r + 1) / 2 + r];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
        (Uplo::Upper, _) => {
            let mut rx = 0;
            for r in 0..n {
                let mut sum = T::zero();
                let mut cx = 0;
                for c in 0..r {
                    sum += ap[r * (r + 1) / 2 + c] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / ap[r * (r + 1) / 2 + r];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, Trans::No) => {
            let mut rx = 0;
            for r in 0..n {
                let col_start_r = r * n - r * (r - 1) / 2;
                let mut sum = T::zero();
                let mut cx = 0;
                for c in 0..r {
                    let cs_c = c * n - c * (c - 1) / 2;
                    sum += ap[cs_c + (r - c)] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / ap[col_start_r];
                }
                rx += incx;
            }
        }
        (Uplo::Lower, _) => {
            let mut rx = (n - 1) * incx;
            for r in (0..n).rev() {
                let col_start_r = r * n - r * (r - 1) / 2;
                let mut sum = T::zero();
                let mut cx = (r + 1) * incx;
                for c in (r + 1)..n {
                    let cs_r = r * n - r * (r - 1) / 2;
                    sum += ap[cs_r + (c - r)] * x[cx];
                    cx += incx;
                }
                x[rx] = x[rx] - sum;
                if nounit {
                    x[rx] = x[rx] / ap[col_start_r];
                }
                if r > 0 {
                    rx -= incx;
                }
            }
        }
    }
}

/// Rank-1 update: A = alpha*x*y^T + A
pub fn ger<T: BlasFloat>(
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) {
    if m == 0 || n == 0 || alpha == T::zero() {
        return;
    }
    let mut jy = 0;
    for j in 0..n {
        let yj = alpha * y[jy];
        let mut ix = 0;
        for i in 0..m {
            a[j * lda + i] += x[ix] * yj;
            ix += incx;
        }
        jy += incy;
    }
}

/// Symmetric rank-1 update: A = alpha*x*x^T + A
pub fn syr<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    a: &mut [T],
    lda: usize,
) {
    if n == 0 || alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let mut ix = 0;
                for i in 0..=j {
                    a[j * lda + i] += x[ix] * xj;
                    ix += incx;
                }
                jx += incx;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let mut ix = jx;
                for i in j..n {
                    a[j * lda + i] += x[ix] * xj;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }
}

/// Symmetric packed rank-1 update: A = alpha*x*x^T + A
pub fn spr<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    ap: &mut [T],
) {
    if n == 0 || alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let mut ix = 0;
                for i in 0..=j {
                    ap[j * (j + 1) / 2 + i] += x[ix] * xj;
                    ix += incx;
                }
                jx += incx;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            for j in 0..n {
                let col_start = j * n - j * (j - 1) / 2;
                let xj = alpha * x[jx];
                let mut ix = jx;
                for i in j..n {
                    ap[col_start + (i - j)] += x[ix] * xj;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }
}

/// Symmetric rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
pub fn syr2<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) {
    if n == 0 || alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let yj = alpha * y[jy];
                let mut ix = 0;
                let mut iy = 0;
                for i in 0..=j {
                    a[j * lda + i] += x[ix] * yj + y[iy] * xj;
                    ix += incx;
                    iy += incy;
                }
                jx += incx;
                jy += incy;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let yj = alpha * y[jy];
                let mut ix = jx;
                let mut iy = jy;
                for i in j..n {
                    a[j * lda + i] += x[ix] * yj + y[iy] * xj;
                    ix += incx;
                    iy += incy;
                }
                jx += incx;
                jy += incy;
            }
        }
    }
}

/// Symmetric packed rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
pub fn spr2<T: BlasFloat>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    ap: &mut [T],
) {
    if n == 0 || alpha == T::zero() {
        return;
    }
    match uplo {
        Uplo::Upper => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let xj = alpha * x[jx];
                let yj = alpha * y[jy];
                let mut ix = 0;
                let mut iy = 0;
                for i in 0..=j {
                    ap[j * (j + 1) / 2 + i] += x[ix] * yj + y[iy] * xj;
                    ix += incx;
                    iy += incy;
                }
                jx += incx;
                jy += incy;
            }
        }
        Uplo::Lower => {
            let mut jx = 0;
            let mut jy = 0;
            for j in 0..n {
                let col_start = j * n - j * (j - 1) / 2;
                let xj = alpha * x[jx];
                let yj = alpha * y[jy];
                let mut ix = jx;
                let mut iy = jy;
                for i in j..n {
                    ap[col_start + (i - j)] += x[ix] * yj + y[iy] * xj;
                    ix += incx;
                    iy += incy;
                }
                jx += incx;
                jy += incy;
            }
        }
    }
}
