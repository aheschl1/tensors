use std::{net::{IpAddr, TcpStream}, sync::{atomic::AtomicBool, Arc}};

use crate::{backend::{remote::TensorId, Backend}, core::value::TensorValue};


struct RemoteCtx {
    pub address: IpAddr,
    pub port: u16
}

pub struct RemoteBuf {
    pub id: TensorId,
    dirty: Arc<AtomicBool>,
}

pub struct RemoteBackend {
    ctx: RemoteCtx,
    stream: Option<TcpStream>
}

impl RemoteBackend {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            ctx: RemoteCtx { address, port },
            stream: None,
        }
    }

    pub fn connect(&self) -> Result<(), std::io::Error> {
        todo!()
    }
}

impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf;

    fn device_type() -> crate::core::primitives::DeviceType {
        todo!()
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        todo!()
    }

    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        todo!()
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        todo!()
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        todo!()
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        todo!()
    }

    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
        todo!()
    }

    fn new() -> Self {
        todo!()
    }

    fn apply_elementwise_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn apply_elementwise_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn apply_elementwise_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (crate::ops::base::OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    unsafe fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &crate::core::MetaTensor), 
        right: (*const Self::Buf<T>, &crate::core::MetaTensor),
        dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
        op: crate::ops::base::OpType
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }
}