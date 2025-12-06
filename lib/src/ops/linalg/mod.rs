use crate::{backend::Backend, core::{primitives::TensorBase, tensor::TensorError, value::TensorValue}};

mod matmul;

pub trait MatMul<Rhs, T, B> 
where 
    T: TensorValue,
    B: Backend<T>,
{
    fn matmul(&self, rhs: &Rhs) -> Result<TensorBase<T, B>, TensorError>;
}