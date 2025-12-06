use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsView, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape}, ops::linalg::MatMul};


impl<L, R, T, B> MatMul<R, T, B> for L
where
    T: TensorValue,
    B: Backend<T>,
    L: AsView<T, B>,
    R: AsView<T, B>,
{
    fn matmul(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError> {
        let view = self.view();
        let rhs_view = rhs.view();

        let mut _copy_left = None;  // just to keep ownership
        let mut _copy_right = None;

        let lhs = if view.is_contiguous() {
            (view.meta, view.buf)
        } else{
            _copy_left = Some(view.contiguous());
            (_copy_left.as_ref().unwrap().meta.clone(), &_copy_left.as_ref().unwrap().buf)
        };

        let rhs = if rhs_view.is_contiguous() {
            (rhs_view.meta, rhs_view.buf)
        } else{
            _copy_right = Some(rhs_view.contiguous());
            (_copy_right.as_ref().unwrap().meta.clone(), &_copy_right.as_ref().unwrap().buf)
        };
        
        let (rshape, rstride) = get_matmul_params(&lhs.0, &rhs.0)?;

        // now we have contiguous lhs and rhs
        
        // let res = TensorBase::<T, B>::zeros(shape)

        panic!()
    }
}

fn get_matmul_params(
    lhs_meta: &MetaTensor,
    rhs_meta: &MetaTensor,
) -> Result<(Shape, Vec<usize>), TensorError> {
    // check dimensions
    if lhs_meta.rank() < 2 || rhs_meta.rank() < 2 {
        return Err(TensorError::InvalidShape);
    }

    let squashed_left_shape = lhs_meta.shape.squash_leading_dims(lhs_meta.rank() - 2);
    let squashed_right_shape = rhs_meta.shape.squash_leading_dims(rhs_meta.rank() - 2);

    let squashed_left_stride = lhs_meta.strides.squash_leading_dims(lhs_meta.rank() - 2);
    let squashed_right_stride = rhs_meta.strides.squash_leading_dims(rhs_meta.rank() - 2);

    panic!()
}
