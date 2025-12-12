
use std::{collections::HashMap, io::Read, net::{IpAddr, TcpListener, TcpStream}, thread};

use crate::{backend::{cpu::Cpu, remote::TensorId, Backend}, core::{tensor::{AsView, AsViewMut, TensorError}, untyped::UntypedTensor, value::TensorValue, Shape, TensorView, TensorViewMut}};

#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;

use prost::Message;
use super::protocol;


struct RemoteTensor<B: Backend> {
    pub id: TensorId,
    tensor: dyn UntypedTensor<B>
}

impl<T: TensorValue, B: Backend> AsView<T, B> for RemoteTensor<B> {
    fn view(&self) -> TensorView<'_, T, B> { 
        self.tensor.typed::<T>().expect("Failed to downcast tensor").view() 
    }
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> { 
        self.tensor.typed::<T>().expect("Failed to downcast tensor").view_as(shape) 
    }
}

impl<T: TensorValue, B: Backend> AsViewMut<T, B> for RemoteTensor<B> {
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> { 
        self.tensor.typed_mut::<T>().expect("Failed to downcast tensor").view_mut() 
    }
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError> { 
        self.tensor.typed_mut::<T>().expect("Failed to downcast tensor").view_as_mut(shape) 
    }
}

struct BackendServer {
    address: IpAddr,
    port: u16,
    handles: HashMap<std::net::SocketAddr, thread::JoinHandle<()>>,
}

impl BackendServer {
    fn new(address: IpAddr, port: u16) -> Self {
        Self { address, port, handles: HashMap::new() }
    }

    fn serve(&mut self) -> Result<(), std::io::Error> {
        let listener = TcpListener::bind((self.address, self.port))?;
        for stream in listener.incoming() {
            let stream = stream?;
            let peer_addr = stream.peer_addr()?;
            let mut handle = ConnectionHandle::new(stream);
            let handle_thread = thread::spawn(move || {
                handle.serve();
            });
            self.handles.insert(peer_addr, handle_thread);
        }
        Ok(())
    }
}


struct ConnectionHandle {
    stream: TcpStream,
    tensors_cpu: HashMap<TensorId, Box<dyn UntypedTensor<Cpu>>>,
    #[cfg(feature = "cuda")]
    tensors_cuda: HashMap<TensorId, Box<dyn UntypedTensor<Cuda>>>,
}

impl ConnectionHandle {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            tensors_cpu: HashMap::new(),
            #[cfg(feature = "cuda")]
            tensors_cuda: HashMap::new(),
        }
    }

    fn serve (&mut self) {
        loop {
            let mut num_bytes = [0u8; 4];
            self.stream.read_exact(&mut num_bytes).expect("Add better failure handling");
            let msg_len = u32::from_le_bytes(num_bytes) as usize;

            let mut buf = vec![0u8; msg_len];
            self.stream.read_exact(&mut buf).expect("Add better failure handling");

            let request = match protocol::BackendRequest::decode(&buf[..]) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("Failed to decode protobuf message: {}", e);
                    continue;
                }
            };

            let response = self.handle_request(request);

        }
    }

    fn handle_request(&mut self, request: protocol::BackendRequest) -> protocol::BackendResponse {
        match request.request {
            Some(protocol::backend_request::Request::AllocFromSlice(req)) => {
                // Handle AllocFromSlice request
            }
            Some(protocol::backend_request::Request::Alloc(req)) => {
                // Handle Alloc request
            }
            _ => {
                eprintln!("Unknown request type");
            }
        }

        todo!()
    }
}