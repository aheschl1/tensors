use std::collections::HashMap;

use crate::{backend::{remote::TensorId, Backend}, core::{primitives::TensorBase, tensor::{AsView, AsViewMut, TensorError}, value::TensorValue, Shape, TensorView, TensorViewMut}};


struct RemoteTensor<T: TensorValue, B: Backend<T>> {
    pub id: TensorId,
    tensor: TensorBase<T, B>,
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for RemoteTensor<T, B> {
    fn view(&self) -> TensorView<'_, T, B> { self.tensor.view() }
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> { self.tensor.view_as(shape) }
}

impl<T: TensorValue, B: Backend<T>> AsViewMut<T, B> for RemoteTensor<T, B> {
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> { self.tensor.view_mut() }
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError> { self.tensor.view_as_mut(shape)}
}

struct TensorHook<T: TensorValue, B: Backend<T>> {
    tensor: RemoteTensor<T, B>,
}