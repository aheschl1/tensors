use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape, TensorView, TensorViewMut}};


pub enum ReductionOpTypes {
    Sum = 1,
    Prod = 2,
    Max = 3,
    Min = 4,
    Mean = 5,
    PopVariance = 6,
    UnbiasedVariance = 7
}

pub trait TotalReductionOp<T: TensorValue, B: Backend>: Sized + ReductionOp<T, B> {
    fn total_sum(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.sum(&Idx::Item)
    }
    fn total_prod(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.prod(&Idx::Item)
    }
    fn total_mean(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.mean(&Idx::Item)
    }
    fn total_max(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.max(&Idx::Item)
    }
    fn total_min(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.min(&Idx::Item)
    }
    fn total_var(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.var(&Idx::Item)
    }
    fn total_pop_var(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.pop_var(&Idx::Item)
    }
}

pub trait ReductionOp<T: TensorValue, B: Backend> : Sized {
    fn sum(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn prod(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn mean(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn max(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn min(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn var(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
    fn pop_var(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>;
}

impl<T: TensorValue, B: Backend, V> TotalReductionOp<T, B> for V where V: ReductionOp<T, B>{}

macro_rules! do_reduce {
    ($op:expr, $axes:ident, $tensor:ident) => {

        match $axes {
            Idx::Item => {
                let mut output = TensorBase::from_buf(vec![ T::ZERO ], vec![])?;
                $tensor.backend.apply_reduce_contiguous_flat(
                    &$tensor.buf,
                    &mut output.buf, 
                    $tensor.meta.offset,
                    $tensor.meta.size(),
                    $op,
                )?;
                Ok(output)
                // Err(TensorError::WrongDims("test".to_string()))
            }
            Idx::At(axis) => {
                let mut output = materialize_output::<T, B>(&$tensor.meta, $tensor.backend.clone(), $axes)?;
                $tensor.backend.apply_reduce(
                    (&$tensor.buf, &$tensor.meta), 
                    (&mut output.buf, &output.meta), 
                    *axis,
                    $op,
                )?;
                Ok(output)
            }
            _ => Err(TensorError::WrongDims(
                "Reduction over multiple axes is not implemented yet.".to_string(),
            ))
        }

      
    };
}

impl<T: TensorValue, B: Backend, V> ReductionOp<T, B> for V
where
    V: AsView<T, B>,
{
    fn sum(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::Sum, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::Sum, axes, t)
        }
    }

    fn prod(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::Prod, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::Prod, axes, t)
        }
    }

    fn max(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::Max, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::Max, axes, t)
        }
    }

    fn min(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::Min, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::Min, axes, t)
        }
    }

    fn mean(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::Mean, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::Mean, axes, t)
        }
    }

    fn var(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::UnbiasedVariance, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::UnbiasedVariance, axes, t)
        }
    }

    fn pop_var(&self, axes: &Idx) -> Result<TensorBase<T, B>, TensorError> {
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce!(ReductionOpTypes::PopVariance, axes, a)
        } else {
            do_reduce!(ReductionOpTypes::PopVariance, axes, t)
        }
    }
}


// implement_reductionop!(TensorBase);
// implement_reductionop!(TensorView<'a>);
// implement_reductionop!(TensorViewMut<'a>);

#[inline]
fn materialize_output<T: TensorValue, B: Backend>(input: &MetaTensor, backend: B, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>{
    let output_meta = reduction_output_meta(input.clone(), axes);
    let buf = backend.alloc(output_meta.size());

   
    Ok(TensorBase::from_parts(backend, buf?, output_meta))
}

#[inline]
fn reduction_output_meta(input: MetaTensor, axes: &Idx) -> MetaTensor {
    let mut output_shape = Vec::new();


    let idx_num = match axes {
        Idx::Item => 0,
        Idx::At(x) => *x,
        _ => panic!("Weird multi dimension reductions are not avaiable.")
    };




    for d in 0..input.rank() {
        if d == idx_num {
            // PyTorch and Numpy have a keep dim argument that removes this.
            output_shape.push(1);
        } else {
            output_shape.push(input.shape()[d]);

        }
    }

    println!("materialziing output: {:?}", output_shape);
    let output_shape: Shape = Shape::from(output_shape);
    let strides = shape_to_stride(&output_shape);

    MetaTensor {
        shape: output_shape,
        strides,
        offset: 0
    }
}