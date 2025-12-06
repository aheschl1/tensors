
use crate::{backend::Backend, core::{meta::TensorOffsetIterator, tensor::TensorError, value::TensorValue, MetaTensor}, ops::base::OpType};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Cpu;

impl<T: TensorValue> Backend<T> for Cpu {
    type Buf = Box<[T]>;



    fn alloc(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch);
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError> {
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds
        )?)
    }

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds),
        }
    }
    
    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
        Ok(src)
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }


        fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for item in bufptr.iter_mut().skip(start).take(len) {
            *item = op.0.apply(*item, op.1);
        }
        Ok(())
    }
    
    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut idx: isize = offset as isize;
        for _ in 0..len {
            bufptr[idx as usize] = op.0.apply(bufptr[idx as usize], op.1);
            idx += stride;
        }
        Ok(())
    }
    
    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: (OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let iterator = TensorOffsetIterator::new(
            shape,
            stride,
            offset,
        );
        for idx in iterator {
            bufptr[idx] = op.0.apply(bufptr[idx], op.1);
        }
        Ok(())
    }

    unsafe fn broadcast(
        &self, 
        left: (*const Self::Buf, &MetaTensor), 
        right: (*const Self::Buf, &MetaTensor),
        dst: (*mut Self::Buf, &MetaTensor),
        op: OpType
    ) -> Result<(), TensorError> {
        // this is a stupid algorithm which is O(rank*size)
        // it can be optimized to O(size) later
        // a cleaner O(rank*size) algorithm just uses the coordinate iterator
        // and converts the, to full offsets
        let (left_buf, left_meta) = left;
        let (right_buf, right_meta) = right;
        let (dst_buf, dst_meta) = dst;

        let rank = dst_meta.rank();

        let sl = left_meta.strides();
        let sr = right_meta.strides();
        let sd = dst_meta.strides();

        
        let mut ol = left_meta.offset() as isize;
        let mut or = right_meta.offset() as isize;
        let mut od = dst_meta.offset() as isize;

        // println!("Strides: left: {:?}, right: {:?}, dst: {:?}", sl, sr, sd);
        // println!("Offsets: left: {}, right: {}, dst: {}", ol, or, od);

        let mut coords = vec![0; rank];

        let mut first = true;

        for new_coord in dst_meta.iter_coords() {
            // println!("Coords: {:?}", new_coord);
            if first {
                first = false;
            } else{
                for d in (0..rank).rev() {
                    if new_coord[d] != coords[d] {
                        let delta = new_coord[d] as isize - coords[d] as isize;
                        ol += delta * sl[d];
                        or += delta * sr[d];
                        od += delta * sd[d];
                    }
                }
            }
            coords = new_coord;
            debug_assert!(od >= 0);
            debug_assert!(ol >= 0);
            debug_assert!(or >= 0);
            // dst_buf[od as usize] = op.apply(left_buf[ol as usize], right_buf[or as usize]);
            unsafe {
                let lval = (*left_buf)[ol as usize];
                let rval = (*right_buf)[or as usize];
                (*dst_buf)[od as usize] = op.apply(lval, rval);
            }
        }

        Ok(())
    }
    
    fn matmul(
        &self,
        lhs: (MetaTensor, &Self::Buf), 
        rhs: (MetaTensor, &Self::Buf)
    ) -> Result<Self::Buf, TensorError> {
        // ugly placeholder
        let (lhs_meta, lhs_buf) = lhs;
        let (rhs_meta, rhs_buf) = rhs;

        if lhs_meta.rank() != rhs_meta.rank() {
            return Err(TensorError::InvalidShape);
        }
        if lhs_meta.rank() < 2{
            return Err(TensorError::InvalidShape);
        }
        if lhs_meta.rank() > 3{
            return Err(TensorError::InvalidShape);
        }


        let batched = lhs_meta.rank() == 3;

        if batched {
            let b = lhs_meta.shape[0]; // batch
            let n = lhs_meta.shape[1]; // rows
            let m = lhs_meta.shape[2]; // cols

            if rhs_meta.shape[0] != b {
                return Err(TensorError::SizeMismatch);
            }
            let p = rhs_meta.shape[1]; // cols
            let q = rhs_meta.shape[2]; // rows
            
            if m != p {
                return Err(TensorError::SizeMismatch);
            }

            //
            //R^{n x m} * R^{m x q} = R^{n x q}
            //

            let mut dst_buf: Box<[T]> = self.alloc(b * n * q)?;
            for batch in 0..b {
                for row in 0..n {
                    for col in 0..q {
                        let mut acc = T::zero();
                        for k in 0..m {
                            let lhs_idx = batch * n * m + row * m + k;
                            let rhs_idx = batch * p * q + k * q + col;
                            unsafe {
                                let lval = lhs_buf.get_unchecked(lhs_idx);
                                let rval = rhs_buf.get_unchecked(rhs_idx);
                                acc = acc + (*lval) * (*rval);
                            }
                        }
                        let dst_idx = batch * n * q + row * q + col;
                        unsafe {
                            let slot = dst_buf.get_unchecked_mut(dst_idx);
                            *slot = acc;
                        }
                    }
                }
            }

            Ok(dst_buf)
        }else{
            Err(TensorError::InvalidShape)
        }
    }
}