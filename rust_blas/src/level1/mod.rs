use crate::types::BlasFloat;

/// Sum of absolute values
#[must_use]
#[inline]
pub fn asum<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> T {
    if n == 0 || incx == 0 {
        return T::zero();
    }
    x.iter().step_by(incx).take(n).fold(T::zero(), |acc, &v| acc + v.abs())
}

/// y = alpha*x + y
#[inline]
pub fn axpy<T: BlasFloat>(n: usize, alpha: T, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let xs = x.iter().step_by(incx).take(n);
    let ys = y.iter_mut().step_by(incy).take(n);
    for (&xi, yi) in xs.zip(ys) {
        *yi += alpha * xi;
    }
}

/// Copy x into y
#[inline]
pub fn copy<T: BlasFloat>(n: usize, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let xs = x.iter().step_by(incx).take(n);
    let ys = y.iter_mut().step_by(incy).take(n);
    for (&xi, yi) in xs.zip(ys) {
        *yi = xi;
    }
}

/// Dot product
#[must_use]
#[inline]
pub fn dot<T: BlasFloat>(n: usize, x: &[T], incx: usize, y: &[T], incy: usize) -> T {
    if n == 0 {
        return T::zero();
    }
    x.iter().step_by(incx).take(n)
        .zip(y.iter().step_by(incy).take(n))
        .fold(T::zero(), |acc, (&xi, &yi)| acc + xi * yi)
}

/// Euclidean norm (scaled algorithm to avoid overflow)
#[must_use]
#[inline]
pub fn nrm2<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> T {
    if n == 0 {
        return T::zero();
    }
    let mut scale = T::zero();
    let mut ssq = T::one();
    for &xi in x.iter().step_by(incx).take(n) {
        let absxi = xi.abs();
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
    }
    scale * ssq.sqrt()
}

/// Scale vector by alpha
#[inline]
pub fn scal<T: BlasFloat>(n: usize, alpha: T, x: &mut [T], incx: usize) {
    if n == 0 {
        return;
    }
    x.iter_mut().step_by(incx).take(n).for_each(|xi| *xi *= alpha);
}

/// Swap x and y
#[inline]
pub fn swap<T: BlasFloat>(n: usize, x: &mut [T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 {
        return;
    }
    let xs = x.iter_mut().step_by(incx).take(n);
    let ys = y.iter_mut().step_by(incy).take(n);
    for (xi, yi) in xs.zip(ys) {
        std::mem::swap(xi, yi);
    }
}

/// Index of element with maximum absolute value (0-based)
#[must_use]
#[inline]
pub fn amax<T: BlasFloat>(n: usize, x: &[T], incx: usize) -> usize {
    if n == 0 {
        return 0;
    }
    x.iter().step_by(incx).take(n)
        .enumerate()
        .fold((0usize, T::zero()), |(max_i, max_v), (i, &v)| {
            let av = v.abs();
            if av > max_v { (i, av) } else { (max_i, max_v) }
        })
        .0
}

/// Givens rotation construction
#[inline]
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
#[inline]
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
    let xs = x.iter_mut().step_by(incx).take(n);
    let ys = y.iter_mut().step_by(incy).take(n);
    for (xi, yi) in xs.zip(ys) {
        let tmp = c * *xi + s * *yi;
        *yi = c * *yi - s * *xi;
        *xi = tmp;
    }
}

// Flag values for the modified Givens rotation matrix stored in param[0]:
//  -2 → identity (H = I, param[1..4] unused)
//  -1 → general H (all four h11,h21,h12,h22 are set in param)
//   0 → h11=h22=1 (only h12,h21 vary)
//   1 → h12=-1, h21=1 (only h11,h22 vary)
const FLAG_IDENTITY: f64  = -2.0;
const FLAG_GENERAL:  f64  = -1.0;
const FLAG_H_OFF_DIAG: f64 = 0.0;
const FLAG_H_DIAG: f64    =  1.0;

/// Modified Givens rotation construction
#[allow(unused_assignments)] // h11/h12/h21/h22 init is conditional by design
#[inline]
pub fn rotmg<T: BlasFloat>(d1: &mut T, d2: &mut T, x1: &mut T, y1: T, param: &mut [T; 5]) {
    let gam    = T::from_f64(4096.0);
    let gamsq  = T::from_f64(4096.0 * 4096.0);
    let rgamsq = T::from_f64(1.0 / (4096.0 * 4096.0));

    let flag_identity   = T::from_f64(FLAG_IDENTITY);
    let flag_general    = T::from_f64(FLAG_GENERAL);
    let flag_h_off_diag = T::from_f64(FLAG_H_OFF_DIAG);
    let flag_h_diag     = T::from_f64(FLAG_H_DIAG);

    if *d1 < T::zero() {
        param[0] = flag_general;
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
        param[0] = flag_identity;
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
            param[0] = flag_general;
            param[1] = T::zero();
            param[2] = T::zero();
            param[3] = T::zero();
            param[4] = T::zero();
            *d1 = T::zero();
            *d2 = T::zero();
            *x1 = T::zero();
            return;
        }
        flag = flag_h_off_diag;
        *d1 /= u;
        *d2 /= u;
        *x1 *= u;
    } else {
        if q2 < T::zero() {
            param[0] = flag_general;
            param[1] = T::zero();
            param[2] = T::zero();
            param[3] = T::zero();
            param[4] = T::zero();
            *d1 = T::zero();
            *d2 = T::zero();
            *x1 = T::zero();
            return;
        }
        flag = flag_h_diag;
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
        if flag == flag_h_off_diag {
            h11 = T::one();
            h22 = T::one();
            flag = flag_general;
        } else {
            h12 = T::one();
            h21 = -T::one();
            flag = flag_general;
        }
        if *d1 <= rgamsq {
            *d1 *= gam * gam;
            *x1 /= gam;
            h11 /= gam;
            h12 /= gam;
        } else {
            *d1 /= gam * gam;
            *x1 *= gam;
            h11 *= gam;
            h12 *= gam;
        }
    }
    while *d2 <= rgamsq || *d2 >= gamsq {
        if flag == flag_h_off_diag {
            h11 = T::one();
            h22 = T::one();
            flag = flag_general;
        } else {
            h12 = T::one();
            h21 = -T::one();
            flag = flag_general;
        }
        if *d2 <= rgamsq {
            *d2 *= gam * gam;
            h21 /= gam;
            h22 /= gam;
        } else {
            *d2 /= gam * gam;
            h21 *= gam;
            h22 *= gam;
        }
    }

    param[0] = flag;
    if flag == flag_general {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if flag == flag_h_off_diag {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
}

/// Apply modified Givens rotation
#[inline]
pub fn rotm<T: BlasFloat>(
    n: usize,
    x: &mut [T],
    incx: usize,
    y: &mut [T],
    incy: usize,
    param: &[T; 5],
) {
    let flag = param[0];
    if n == 0 || flag == T::from_f64(FLAG_IDENTITY) {
        return;
    }
    let (h11, h21, h12, h22) = (param[1], param[2], param[3], param[4]);

    if flag == T::from_f64(FLAG_GENERAL) {
        for (xi, yi) in x.iter_mut().step_by(incx).take(n).zip(y.iter_mut().step_by(incy).take(n)) {
            let (w, z) = (*xi, *yi);
            *xi = w * h11 + z * h12;
            *yi = w * h21 + z * h22;
        }
    } else if flag == T::from_f64(FLAG_H_OFF_DIAG) {
        for (xi, yi) in x.iter_mut().step_by(incx).take(n).zip(y.iter_mut().step_by(incy).take(n)) {
            let (w, z) = (*xi, *yi);
            *xi = w + z * h12;
            *yi = w * h21 + z;
        }
    } else {
        for (xi, yi) in x.iter_mut().step_by(incx).take(n).zip(y.iter_mut().step_by(incy).take(n)) {
            let (w, z) = (*xi, *yi);
            *xi = w * h11 + z;
            *yi = -w + z * h22;
        }
    }
}

/// Dot product of f32 vectors accumulated in f64
#[must_use]
#[inline]
pub fn dsdot(n: usize, sx: &[f32], incx: usize, sy: &[f32], incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sx.iter().step_by(incx).take(n)
        .zip(sy.iter().step_by(incy).take(n))
        .fold(0.0f64, |acc, (&xi, &yi)| acc + xi as f64 * yi as f64)
}

/// Dot product of f32 vectors with f32 bias, accumulated in f64, returned as f32
#[must_use]
#[inline]
pub fn sdsdot(n: usize, sb: f32, sx: &[f32], incx: usize, sy: &[f32], incy: usize) -> f32 {
    (sb as f64 + dsdot(n, sx, incx, sy, incy)) as f32
}
