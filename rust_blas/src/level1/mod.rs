use crate::types::BlasFloat;

/// Sum of absolute values
pub fn asum<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> T {
    if n == 0 || incx == 0 {
        return T::zero();
    }
    let mut sum = T::zero();
    let mut i = 0;
    for _ in 0..n {
        sum += x[i].abs();
        i += incx;
    }
    sum
}

/// y = alpha*x + y
pub fn axpy<T: BlasFloat>(n: usize, alpha: T, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}

/// Copy x into y
pub fn copy<T: BlasFloat>(n: usize, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = x[ix];
        ix += incx;
        iy += incy;
    }
}

/// Dot product
pub fn dot<T: BlasFloat>(n: usize, x: &[T], incx: usize, y: &[T], incy: usize) -> T {
    if n == 0 {
        return T::zero();
    }
    let mut sum = T::zero();
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        sum += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    sum
}

/// Euclidean norm (scaled algorithm to avoid overflow)
pub fn nrm2<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> T {
    if n == 0 {
        return T::zero();
    }
    let mut scale = T::zero();
    let mut ssq = T::one();
    let mut ix = 0;
    for _ in 0..n {
        let absxi = x[ix].abs();
        if absxi != T::zero() {
            if scale < absxi {
                let ratio = scale / absxi;
                ssq = T::one() + ssq * ratio * ratio;
                scale = absxi;
            } else {
                let ratio = absxi / scale;
                ssq += ratio * ratio;
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// Scale vector by alpha
pub fn scal<T: BlasFloat>(n: usize, alpha: T, x: &mut [T], incx: usize) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    for _ in 0..n {
        x[ix] *= alpha;
        ix += incx;
    }
}

/// Swap x and y
pub fn swap<T: BlasFloat>(n: usize, x: &mut [T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        let tmp = x[ix];
        x[ix] = y[iy];
        y[iy] = tmp;
        ix += incx;
        iy += incy;
    }
}

/// Index of element with maximum absolute value (0-based)
pub fn amax<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut max_val = x[0].abs();
    let mut max_idx = 0;
    let mut ix = incx;
    for i in 1..n {
        let val = x[ix].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx
}

/// Givens rotation construction
pub fn rotg<T: BlasFloat>(a: &mut T, b: &mut T, c: &mut T, s: &mut T) {
    let aa = a.abs();
    let ba = b.abs();
    let roe = if aa > ba { *a } else { *b };
    let scale = aa + ba;
    if scale == T::zero() {
        *c = T::one();
        *s = T::zero();
        *a = T::zero();
        *b = T::zero();
    } else {
        let sa = *a / scale;
        let sb = *b / scale;
        let r = scale * (sa * sa + sb * sb).sqrt();
        let r = if roe.is_negative() { -r } else { r };
        *c = *a / r;
        *s = *b / r;
        let z = if aa > ba {
            *s
        } else if *c != T::zero() {
            T::one() / *c
        } else {
            T::one()
        };
        *a = r;
        *b = z;
    }
}

/// Apply Givens rotation
pub fn rot<T: BlasFloat>(
    n: usize,
    x: &mut [T],
    incx: usize,
    y: &mut [T],
    incy: usize,
    c: T,
    s: T,
) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        let tmp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = tmp;
        ix += incx;
        iy += incy;
    }
}

/// Modified Givens rotation construction
pub fn rotmg<T: BlasFloat>(d1: &mut T, d2: &mut T, x1: &mut T, y1: T, param: &mut [T; 5]) {
    let gam = T::from_f64(4096.0);
    let gamsq = T::from_f64(4096.0 * 4096.0);
    let rgamsq = T::from_f64(1.0 / (4096.0 * 4096.0));

    if *d1 < T::zero() {
        param[0] = T::from_f64(-1.0);
        param[1] = T::zero();
        param[2] = T::zero();
        param[3] = T::zero();
        param[4] = T::zero();
        *d1 = T::zero();
        *d2 = T::zero();
        *x1 = T::zero();
        return;
    }
    let p2 = *d2 * y1;
    if p2 == T::zero() {
        param[0] = T::from_f64(-2.0);
        return;
    }
    let p1 = *d1 * *x1;
    let q2 = p2 * y1;
    let q1 = p1 * *x1;

    let mut flag;
    let mut h11 = T::zero();
    let mut h12 = T::zero();
    let mut h21 = T::zero();
    let mut h22 = T::zero();

    if q1.abs() > q2.abs() {
        h21 = -y1 / *x1;
        h12 = p2 / p1;
        let u = T::one() - h12 * h21;
        if u <= T::zero() {
            param[0] = T::from_f64(-1.0);
            param[1] = T::zero();
            param[2] = T::zero();
            param[3] = T::zero();
            param[4] = T::zero();
            *d1 = T::zero();
            *d2 = T::zero();
            *x1 = T::zero();
            return;
        }
        flag = T::zero();
        *d1 = *d1 / u;
        *d2 = *d2 / u;
        *x1 = *x1 * u;
    } else {
        if q2 < T::zero() {
            param[0] = T::from_f64(-1.0);
            param[1] = T::zero();
            param[2] = T::zero();
            param[3] = T::zero();
            param[4] = T::zero();
            *d1 = T::zero();
            *d2 = T::zero();
            *x1 = T::zero();
            return;
        }
        flag = T::one();
        h11 = p1 / p2;
        h22 = *x1 / y1;
        let u = T::one() + h11 * h22;
        let d1_new = *d2 / u;
        let d2_new = *d1 / u;
        *d1 = d1_new;
        *d2 = d2_new;
        *x1 = y1 * u;
        h12 = T::one();
        h21 = -T::one();
    }

    // Rescale to avoid overflow/underflow
    while *d1 <= rgamsq || *d1 >= gamsq {
        if flag == T::zero() {
            h11 = T::one();
            h22 = T::one();
            flag = T::from_f64(-1.0);
        } else {
            h12 = T::one();
            h21 = -T::one();
            flag = T::from_f64(-1.0);
        }
        if *d1 <= rgamsq {
            *d1 = *d1 * gam * gam;
            *x1 = *x1 / gam;
            h11 = h11 / gam;
            h12 = h12 / gam;
        } else {
            *d1 = *d1 / (gam * gam);
            *x1 = *x1 * gam;
            h11 = h11 * gam;
            h12 = h12 * gam;
        }
    }
    while *d2 <= rgamsq || *d2 >= gamsq {
        if flag == T::zero() {
            h11 = T::one();
            h22 = T::one();
            flag = T::from_f64(-1.0);
        } else {
            h12 = T::one();
            h21 = -T::one();
            flag = T::from_f64(-1.0);
        }
        if *d2 <= rgamsq {
            *d2 = *d2 * gam * gam;
            h21 = h21 / gam;
            h22 = h22 / gam;
        } else {
            *d2 = *d2 / (gam * gam);
            h21 = h21 * gam;
            h22 = h22 * gam;
        }
    }

    param[0] = flag;
    if flag == T::from_f64(-1.0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if flag == T::zero() {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
}

/// Apply modified Givens rotation
pub fn rotm<T: BlasFloat>(
    n: usize,
    x: &mut [T],
    incx: usize,
    y: &mut [T],
    incy: usize,
    param: &[T; 5],
) {
    let flag = param[0];
    if n == 0 || flag == T::from_f64(-2.0) {
        return;
    }
    let h11 = param[1];
    let h21 = param[2];
    let h12 = param[3];
    let h22 = param[4];
    let mut ix = 0;
    let mut iy = 0;
    if flag == T::from_f64(-1.0) {
        for _ in 0..n {
            let w = x[ix];
            let z = y[iy];
            x[ix] = w * h11 + z * h12;
            y[iy] = w * h21 + z * h22;
            ix += incx;
            iy += incy;
        }
    } else if flag == T::zero() {
        for _ in 0..n {
            let w = x[ix];
            let z = y[iy];
            x[ix] = w + z * h12;
            y[iy] = w * h21 + z;
            ix += incx;
            iy += incy;
        }
    } else {
        for _ in 0..n {
            let w = x[ix];
            let z = y[iy];
            x[ix] = w * h11 + z;
            y[iy] = -w + z * h22;
            ix += incx;
            iy += incy;
        }
    }
}

/// Dot product of f32 vectors accumulated in f64
pub fn dsdot(n: usize, sx: &[f32], incx: usize, sy: &[f32], incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        sum += sx[ix] as f64 * sy[iy] as f64;
        ix += incx;
        iy += incy;
    }
    sum
}

/// Dot product of f32 vectors with f32 bias, accumulated in f64, returned as f32
pub fn sdsdot(n: usize, sb: f32, sx: &[f32], incx: usize, sy: &[f32], incy: usize) -> f32 {
    (sb as f64 + dsdot(n, sx, incx, sy, incy)) as f32
}
