use std::ops::Div;

use crate::core::value::{TensorValue, types::boolean};



pub trait Exp: TensorValue {
    fn apply_exp(&self) -> Self;
}


impl Exp for f32 {
    #[inline]
    fn apply_exp(&self) -> Self {
        self.exp()
    }
}

impl Exp for f64 {
    #[inline]
    fn apply_exp(&self) -> Self {
        self.exp()
    }
}

// TODO: Apply for other types.