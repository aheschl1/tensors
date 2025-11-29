

use crate::{core::{meta::MemRegion, tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::unary::ElementwiseTensorOp};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;

pub trait Backend<T: TensorValue> {
    type Buf;

    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;
    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;
    fn len(&self, buf: &Self::Buf) -> usize;
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError>;
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError>;
    fn new() -> Self;
}

pub trait BackendUnaryElementwise<T: TensorValue + TensorValueElementwise>: Backend<T> {
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;
    fn apply_elementwise_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;
    // fn apply_elementwise_scattered(
    //     &self, buf: &mut Self::Buf, 
    //     op: &ElementwiseTensorOp<T>, 
    //     offsets: &[usize]
    // ) -> Result<(), TensorError>;

    fn apply_elementwise(&self, buf: &mut Self::Buf, op: ElementwiseTensorOp<T>, regions: &[MemRegion]) -> Result<(), TensorError>{
        for region in regions {
            match region {
                MemRegion::Contiguous{start, len} => {
                    self.apply_elementwise_contiguous(buf, &op, *start, *len)?
                },
                MemRegion::Strided{start, stride, len} => {
                    self.apply_elementwise_strided(buf, &op, *start, *stride, *len)?
                },
                // MemRegion::Scattered{offsets} => {
                //     self.apply_elementwise_scattered(buf, &op, offsets)?
                // },
            }
        }
        Ok(())
    }
}

pub trait BackendBinaryElementwise<T: TensorValue + TensorValueElementwise>: Backend<T> {
    /// Rules of the merge operation:
    /// - left and right regions, when combined, cover the same number of indices
    /// - left and right regions, when combined, cover the same number of indices as the dst regions
    /// - dst addresses do not overlap with left or right addresses
    /// 
    /// In general, broadcasting rukes with regard to shape are:
    /// Traversing from most minor to most major dimension:
    /// - if dimensions are equal, they align
    /// - if one dimension is 1, it is broadcast to the other dimension
    /// - if one dimension is missing, it is treated as 1 and broadcast to the other dimension
    /// 
    /// The memory layouts can be arbitrary, as long as the above rules are followed.
    fn merge(
        &self, 
        left: (&Self::Buf, &[MemRegion]), 
        right: (&Self::Buf, &[MemRegion]),
        dst: (&mut Self::Buf, &[MemRegion]),
        op: ElementwiseTensorOp<T>
    ) -> Result<(), TensorError>;
}