pub trait BlasFloat:
    Copy
    + Default
    + PartialOrd
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
    #[must_use] fn zero() -> Self;
    #[must_use] fn one() -> Self;
    #[must_use] fn two() -> Self;
    #[must_use] fn from_f64(v: f64) -> Self;
    #[must_use] fn abs(self) -> Self;
    #[must_use] fn sqrt(self) -> Self;
    #[must_use] fn signum(self) -> Self;
    #[must_use] fn max(self, other: Self) -> Self;
    #[must_use] fn min(self, other: Self) -> Self;
    #[must_use] fn is_positive(self) -> bool;
    #[must_use] fn is_negative(self) -> bool;
}

impl BlasFloat for f32 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn two() -> Self { 2.0 }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn signum(self) -> Self { self.signum() }
    #[inline] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline] fn is_positive(self) -> bool { self > 0.0 }
    #[inline] fn is_negative(self) -> bool { self < 0.0 }
}

impl BlasFloat for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn two() -> Self { 2.0 }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn signum(self) -> Self { self.signum() }
    #[inline] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline] fn is_positive(self) -> bool { self > 0.0 }
    #[inline] fn is_negative(self) -> bool { self < 0.0 }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Trans { No, Yes, Conj }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Uplo { Upper, Lower }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Diag { NonUnit, Unit }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side { Left, Right }
