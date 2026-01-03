use crate::{backend::Backend, core::{tensor::AsViewMut, value::TensorValue}};

// pub trait Sin<T: TensorValue, B: Backend> {
//     fn sin_inplace(&mut self);
// }

// // pub trait Abs<T: TensorValue, B: Backend> : AsTensor<T, B> {
// //     fn abs(&self) -> TensorBase<T, B>;
// // }

// impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> Sin<T, B> for V {
//     fn sin_inplace(&mut self) {
//         let view = self.view_mut();
//         // will never fail, could in theory ignore the result
//         if let Err(e) = view.backend.apply_sin(view.buf, &view.meta) {
//             panic!("Failed to apply abs: {}", e);
//         }
//     }
// }