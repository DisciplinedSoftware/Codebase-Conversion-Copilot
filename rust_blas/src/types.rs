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
{
    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self;
    fn from_f64(v: f64) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn signum(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
}

impl BlasFloat for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn two() -> Self { 2.0 }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn abs(self) -> Self { f32::abs(self) }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn signum(self) -> Self { f32::signum(self) }
    fn max(self, other: Self) -> Self { if self > other { self } else { other } }
    fn min(self, other: Self) -> Self { if self < other { self } else { other } }
    fn is_positive(self) -> bool { self > 0.0 }
    fn is_negative(self) -> bool { self < 0.0 }
}

impl BlasFloat for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn two() -> Self { 2.0 }
    fn from_f64(v: f64) -> Self { v }
    fn abs(self) -> Self { f64::abs(self) }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn signum(self) -> Self { f64::signum(self) }
    fn max(self, other: Self) -> Self { if self > other { self } else { other } }
    fn min(self, other: Self) -> Self { if self < other { self } else { other } }
    fn is_positive(self) -> bool { self > 0.0 }
    fn is_negative(self) -> bool { self < 0.0 }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Trans { No, Yes, Conj }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Uplo { Upper, Lower }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Diag { NonUnit, Unit }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side { Left, Right }
