pub mod client {
    use std::{
        collections::HashMap, fmt::Debug, io::{Read, Write},
        net::IpAddr,
        sync::{
            atomic::{AtomicBool, AtomicU32, Ordering},
            Arc, Condvar, Mutex, RwLock,
        },
    };
    use crate::{
        backend::{
            remote::{
                get_backend_default,
                protocol::{Messages, Request, Response, Slice, TypelessBuf, Value},
            },
            Backend, BackendMatMul,
        },
        core::{
            primitives::DeviceType, primops::{Exp, InvExp},
            tensor::TensorError, value::{DType, TensorValue},
        },
    };
    use flume;
    pub struct RemoteBuf<T: TensorValue> {
        pub(crate) id: u32,
        pub(crate) dtype: DType,
        pub(crate) _marker: std::marker::PhantomData<T>,
    }
    #[automatically_derived]
    impl<T: ::core::fmt::Debug + TensorValue> ::core::fmt::Debug for RemoteBuf<T> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "RemoteBuf",
                "id",
                &self.id,
                "dtype",
                &self.dtype,
                "_marker",
                &&self._marker,
            )
        }
    }
    #[automatically_derived]
    impl<T: ::core::clone::Clone + TensorValue> ::core::clone::Clone for RemoteBuf<T> {
        #[inline]
        fn clone(&self) -> RemoteBuf<T> {
            RemoteBuf {
                id: ::core::clone::Clone::clone(&self.id),
                dtype: ::core::clone::Clone::clone(&self.dtype),
                _marker: ::core::clone::Clone::clone(&self._marker),
            }
        }
    }
    impl<T: TensorValue> RemoteBuf<T> {
        #[inline(always)]
        fn to_typeless(&self) -> TypelessBuf {
            TypelessBuf {
                id: self.id,
                dtype: self.dtype,
            }
        }
        #[inline(always)]
        pub(crate) fn from_typeless(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData,
            }
        }
    }
    impl<T: TensorValue> From<&mut RemoteBuf<T>> for TypelessBuf {
        fn from(buf: &mut RemoteBuf<T>) -> Self {
            buf.to_typeless()
        }
    }
    impl<T: TensorValue> From<&RemoteBuf<T>> for TypelessBuf {
        fn from(buf: &RemoteBuf<T>) -> Self {
            buf.to_typeless()
        }
    }
    impl<T: TensorValue> From<*const RemoteBuf<T>> for TypelessBuf {
        fn from(buf: *const RemoteBuf<T>) -> Self {
            unsafe { (&*buf).to_typeless() }
        }
    }
    impl<T: TensorValue> From<*mut RemoteBuf<T>> for TypelessBuf {
        fn from(buf: *mut RemoteBuf<T>) -> Self {
            unsafe { (&*buf).to_typeless() }
        }
    }
    struct PendingHandler {
        count: Arc<AtomicU32>,
        cv: Condvar,
        mutex: Mutex<()>,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for PendingHandler {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "PendingHandler",
                "count",
                &self.count,
                "cv",
                &self.cv,
                "mutex",
                &&self.mutex,
            )
        }
    }
    impl PendingHandler {
        fn inc(&self) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        fn dec(&self) {
            let prev = self.count.fetch_sub(1, Ordering::SeqCst);
            if prev == 1 {
                self.cv.notify_all();
            }
        }
        fn sync(&self) {
            while self.count.load(Ordering::SeqCst) > 0 {
                let lock = self.mutex.lock().unwrap();
                let _unused = self.cv.wait(lock).unwrap();
            }
        }
    }
    pub struct RemoteBackend {
        remote_addr: IpAddr,
        remote_port: u16,
        message_id: Arc<AtomicU32>,
        pending: Arc<PendingHandler>,
        messages_outgoing_sender: flume::Sender<Request>,
        messages_outgoing_receiver: flume::Receiver<Request>,
        pending_response: Arc<RwLock<HashMap<u32, flume::Sender<Messages>>>>,
        poisoned: Arc<AtomicBool>,
    }
    #[automatically_derived]
    impl ::core::clone::Clone for RemoteBackend {
        #[inline]
        fn clone(&self) -> RemoteBackend {
            RemoteBackend {
                remote_addr: ::core::clone::Clone::clone(&self.remote_addr),
                remote_port: ::core::clone::Clone::clone(&self.remote_port),
                message_id: ::core::clone::Clone::clone(&self.message_id),
                pending: ::core::clone::Clone::clone(&self.pending),
                messages_outgoing_sender: ::core::clone::Clone::clone(
                    &self.messages_outgoing_sender,
                ),
                messages_outgoing_receiver: ::core::clone::Clone::clone(
                    &self.messages_outgoing_receiver,
                ),
                pending_response: ::core::clone::Clone::clone(&self.pending_response),
                poisoned: ::core::clone::Clone::clone(&self.poisoned),
            }
        }
    }
    impl Debug for RemoteBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RemoteBackend")
                .field("remote_addr", &self.remote_addr)
                .field("remote_port", &self.remote_port)
                .field("message_id", &self.message_id)
                .field("pending", &self.pending)
                .field("pending_response", &self.pending_response)
                .finish()
        }
    }
    impl RemoteBackend {
        pub fn new_with_address(
            remote_addr: IpAddr,
            remote_port: u16,
        ) -> Result<Self, std::io::Error> {
            let pending = PendingHandler {
                count: Arc::new(AtomicU32::new(0)),
                cv: Condvar::new(),
                mutex: Mutex::new(()),
            };
            let (sender, receiver) = flume::unbounded();
            let res = Self {
                remote_addr,
                remote_port,
                pending: Arc::new(pending),
                message_id: Arc::new(AtomicU32::new(0)),
                messages_outgoing_sender: sender,
                messages_outgoing_receiver: receiver,
                pending_response: Arc::new(RwLock::new(HashMap::new())),
                poisoned: Arc::new(AtomicBool::new(false)),
            };
            Ok(res)
        }
        fn poison(&self) {
            self.poisoned.store(true, Ordering::SeqCst);
        }
        pub fn sync(&self) {
            self.pending.sync();
        }
        #[inline(always)]
        fn is_poisoned(&self) -> bool {
            self.poisoned.load(Ordering::SeqCst)
        }
        fn send_message(&self, msg: Messages) -> flume::Receiver<Messages> {
            if self.is_poisoned() {
                {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "Attempted to send message on poisoned RemoteBackend. Reasons for poison:\n            1. An asynchronous operation reported an error from the remote backend.\n            2. The RemoteBackend is in an inconsistent state and can no longer process messages safely.",
                        ),
                    );
                };
            }
            self.pending.inc();
            let (sender, receiver) = flume::bounded(1);
            let mid = self.next_message_id();
            {
                let mut pending = self.pending_response.write().unwrap();
                pending.insert(mid, sender);
            }
            let req = Request {
                task_id: mid,
                message: msg,
            };
            self.messages_outgoing_sender
                .send(req)
                .expect("Failed to send message to outgoing channel");
            receiver
        }
        pub(crate) fn connect(&mut self) -> Result<(), std::io::Error> {
            let stream = std::net::TcpStream::connect((
                self.remote_addr,
                self.remote_port,
            ))?;
            stream.set_nodelay(true)?;
            let read_stream = stream.try_clone()?;
            let write_stream = stream;
            let remote = self.clone();
            std::thread::spawn(move || {
                drain_outgoing(remote, write_stream);
            });
            let remote = self.clone();
            std::thread::spawn(move || {
                read_incoming(remote, read_stream);
            });
            Ok(())
        }
        pub fn address(&self) -> (IpAddr, u16) {
            (self.remote_addr, self.remote_port)
        }
        #[inline(always)]
        fn next_message_id(&self) -> u32 {
            self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl Backend for RemoteBackend {
        type Buf<T: TensorValue> = RemoteBuf<T>;
        fn new() -> Self {
            get_backend_default().expect("No default remote backend available")
        }
        fn device_type() -> crate::core::primitives::DeviceType {
            DeviceType::Remote {
                ip: "127.0.0.1".parse().unwrap(),
                port: 7878,
                remote_type: DeviceType::Cpu.into(),
            }
        }
        fn alloc_from_slice<T: TensorValue>(
            &self,
            src: Box<[T]>,
        ) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
            let message = RpcMessages::AllocFromSlice {
                src: src.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn alloc<T: TensorValue>(
            &self,
            len: usize,
        ) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
            let message = RpcMessages::Alloc {
                len: len.into(),
                dtype: T::DTYPE,
            };
            ::core::panicking::panic("explicit panic")
        }
        fn copy_from_slice<T: TensorValue>(
            &self,
            dst: &mut Self::Buf<T>,
            src: &[T],
        ) -> Result<(), crate::core::tensor::TensorError> {
            let message = RpcMessages::CopyFromSlice {
                dst: dst.into(),
                src: src.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn read<T: TensorValue>(
            &self,
            buf: &Self::Buf<T>,
            offset: usize,
        ) -> Result<T, crate::core::tensor::TensorError> {
            let message = RpcMessages::Read {
                buf: buf.into(),
                offset: offset.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn write<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            value: T,
        ) -> Result<(), crate::core::tensor::TensorError> {
            let message = RpcMessages::Write {
                buf: buf.into(),
                offset: offset.into(),
                value: value.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
            let message = RpcMessages::Len {
                buf: buf.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn copy<T: TensorValue>(
            &self,
            src: &Self::Buf<T>,
        ) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
            let message = RpcMessages::Copy {
                src: src.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn dump<T: TensorValue>(
            &self,
            src: &Self::Buf<T>,
        ) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
            let message = RpcMessages::Dump {
                src: src.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn broadcast<T: TensorValue>(
            &self,
            left: (*const Self::Buf<T>, &crate::core::MetaTensor),
            right: (*const Self::Buf<T>, &crate::core::MetaTensor),
            dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
            op: crate::ops::base::BinaryOpType,
        ) -> Result<(), crate::core::tensor::TensorError> {
            let message = RpcMessages::Broadcast {
                left: (left.0.into(), left.1.into()),
                right: (right.0.into(), right.1.into()),
                dst: (dst.0.into(), dst.1.into()),
                op: op.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_elementwise_binary_contiguous<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            op: (crate::ops::base::BinaryOpType, T),
            start: usize,
            len: usize,
        ) -> Result<(), crate::core::tensor::TensorError> {
            let message = RpcMessages::ApplyElementwiseBinaryContiguous {
                buf: buf.into(),
                op: (op.0.into(), op.1.into()),
                start: start.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_elementwise_binary_1d_strided<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            op: (crate::ops::base::BinaryOpType, T),
            offset: usize,
            stride: isize,
            len: usize,
        ) -> Result<(), crate::core::tensor::TensorError> {
            let message = RpcMessages::ApplyElementwiseBinary1dStrided {
                buf: buf.into(),
                op: (op.0.into(), op.1.into()),
                offset: offset.into(),
                stride: stride.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_elementwise_binary_nd<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            op: (crate::ops::base::BinaryOpType, T),
            offset: usize,
            shape: &[usize],
            stride: &[isize],
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyElementwiseBinaryNd {
                buf: buf.into(),
                op: (op.0.into(), op.1.into()),
                offset: offset.into(),
                shape: shape.into(),
                stride: stride.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_neg_contiguous<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            start: usize,
            len: usize,
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyNegContiguous {
                buf: buf.into(),
                start: start.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_neg_1d_strided<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            stride: isize,
            len: usize,
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyNeg1dStrided {
                buf: buf.into(),
                offset: offset.into(),
                stride: stride.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_neg_nd<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            shape: &[usize],
            stride: &[isize],
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyNegNd {
                buf: buf.into(),
                offset: offset.into(),
                shape: shape.into(),
                stride: stride.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_relu_nd<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            shape: &[usize],
            stride: &[isize],
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyReluNd {
                buf: buf.into(),
                offset: offset.into(),
                shape: shape.into(),
                stride: stride.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_relu_1d_strided<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            stride: isize,
            len: usize,
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyRelu1dStrided {
                buf: buf.into(),
                offset: offset.into(),
                stride: stride.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_relu_contiguous<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            start: usize,
            len: usize,
        ) -> Result<(), TensorError> {
            let message = RpcMessages::ApplyReluContiguous {
                buf: buf.into(),
                start: start.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_sigmoid_nd<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            shape: &[usize],
            stride: &[isize],
        ) -> Result<(), TensorError>
        where
            T: InvExp,
        {
            let message = RpcMessages::ApplySigmoidNd {
                buf: buf.into(),
                offset: offset.into(),
                shape: shape.into(),
                stride: stride.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_sigmoid_1d_strided<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            stride: isize,
            len: usize,
        ) -> Result<(), TensorError>
        where
            T: InvExp,
        {
            let message = RpcMessages::ApplySigmoid1dStrided {
                buf: buf.into(),
                offset: offset.into(),
                stride: stride.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_sigmoid_contiguous<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            start: usize,
            len: usize,
        ) -> Result<(), TensorError>
        where
            T: InvExp,
        {
            let message = RpcMessages::ApplySigmoidContiguous {
                buf: buf.into(),
                start: start.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_tanh_nd<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            shape: &[usize],
            stride: &[isize],
        ) -> Result<(), TensorError>
        where
            T: Exp,
        {
            let message = RpcMessages::ApplyTanhNd {
                buf: buf.into(),
                offset: offset.into(),
                shape: shape.into(),
                stride: stride.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_tanh_1d_strided<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            offset: usize,
            stride: isize,
            len: usize,
        ) -> Result<(), TensorError>
        where
            T: Exp,
        {
            let message = RpcMessages::ApplyTanh1dStrided {
                buf: buf.into(),
                offset: offset.into(),
                stride: stride.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
        fn apply_tanh_contiguous<T: TensorValue>(
            &self,
            buf: &mut Self::Buf<T>,
            start: usize,
            len: usize,
        ) -> Result<(), TensorError>
        where
            T: Exp,
        {
            let message = RpcMessages::ApplyTanhContiguous {
                buf: buf.into(),
                start: start.into(),
                len: len.into(),
            };
            ::core::panicking::panic("explicit panic")
        }
    }
    enum RpcMessages {
        AllocFromSlice { src: Slice },
        AllocFromSliceResponse(Result<TypelessBuf, TensorError>),
        Alloc { len: usize, dtype: DType },
        AllocResponse(Result<TypelessBuf, TensorError>),
        CopyFromSlice { dst: TypelessBuf, src: Slice },
        CopyFromSliceResponse(Result<(), crate::core::tensor::TensorError>),
        Read { buf: TypelessBuf, offset: usize },
        ReadResponse(Result<Value, TensorError>),
        Write { buf: TypelessBuf, offset: usize, value: Value },
        WriteResponse(Result<(), crate::core::tensor::TensorError>),
        Len { buf: TypelessBuf },
        LenResponse(usize),
        Copy { src: TypelessBuf },
        CopyResponse(Result<TypelessBuf, TensorError>),
        Dump { src: TypelessBuf },
        DumpResponse(Result<Slice, crate::core::tensor::TensorError>),
        Broadcast {
            left: (TypelessBuf, crate::core::MetaTensor),
            right: (TypelessBuf, crate::core::MetaTensor),
            dst: (TypelessBuf, crate::core::MetaTensor),
            op: crate::ops::base::BinaryOpType,
        },
        BroadcastResponse(Result<(), crate::core::tensor::TensorError>),
        ApplyElementwiseBinaryContiguous {
            buf: TypelessBuf,
            op: (crate::ops::base::BinaryOpType, Value),
            start: usize,
            len: usize,
        },
        ApplyElementwiseBinaryContiguousResponse(
            Result<(), crate::core::tensor::TensorError>,
        ),
        ApplyElementwiseBinary1dStrided {
            buf: TypelessBuf,
            op: (crate::ops::base::BinaryOpType, Value),
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyElementwiseBinary1dStridedResponse(
            Result<(), crate::core::tensor::TensorError>,
        ),
        ApplyElementwiseBinaryNd {
            buf: TypelessBuf,
            op: (crate::ops::base::BinaryOpType, Value),
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyElementwiseBinaryNdResponse(Result<(), TensorError>),
        ApplyNegContiguous { buf: TypelessBuf, start: usize, len: usize },
        ApplyNegContiguousResponse(Result<(), TensorError>),
        ApplyNeg1dStrided { buf: TypelessBuf, offset: usize, stride: isize, len: usize },
        ApplyNeg1dStridedResponse(Result<(), TensorError>),
        ApplyNegNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyNegNdResponse(Result<(), TensorError>),
        ApplyReluNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyReluNdResponse(Result<(), TensorError>),
        ApplyRelu1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyRelu1dStridedResponse(Result<(), TensorError>),
        ApplyReluContiguous { buf: TypelessBuf, start: usize, len: usize },
        ApplyReluContiguousResponse(Result<(), TensorError>),
        ApplySigmoidNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplySigmoidNdResponse(Result<(), TensorError>),
        ApplySigmoid1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplySigmoid1dStridedResponse(Result<(), TensorError>),
        ApplySigmoidContiguous { buf: TypelessBuf, start: usize, len: usize },
        ApplySigmoidContiguousResponse(Result<(), TensorError>),
        ApplyTanhNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyTanhNdResponse(Result<(), TensorError>),
        ApplyTanh1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyTanh1dStridedResponse(Result<(), TensorError>),
        ApplyTanhContiguous { buf: TypelessBuf, start: usize, len: usize },
        ApplyTanhContiguousResponse(Result<(), TensorError>),
    }
    impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
        fn matmul(
            &self,
            lhs: (&Self::Buf<T>, &crate::core::MetaTensor),
            rhs: (&Self::Buf<T>, &crate::core::MetaTensor),
            dst: &mut Self::Buf<T>,
            b: usize,
            m: usize,
            k: usize,
            n: usize,
            contiguity: crate::core::meta::ContiguityTypes,
        ) -> Result<(), TensorError> {
            let message: Messages = Messages::Matmul {
                lhs: (lhs.0.to_typeless(), lhs.1.clone()),
                rhs: (rhs.0.to_typeless(), rhs.1.clone()),
                dst: dst.to_typeless(),
                b,
                m,
                k,
                n,
                contiguity,
            };
            {
                let receiver = self.send_message(message);
                let response = receiver
                    .recv()
                    .map_err(|_| TensorError::BackendError(
                        "Failed to receive response".to_string(),
                    ))?;
                match response {
                    Messages::MatmulResponse { result } => result,
                    _ => {
                        Err(
                            TensorError::BackendError(
                                "Unexpected response type".to_string(),
                            ),
                        )
                    }
                }
            }
        }
    }
    #[inline]
    fn read_response(
        stream: &mut std::net::TcpStream,
        len_buf: &mut [u8; 4],
    ) -> Result<Response, Box<bincode::ErrorKind>> {
        stream.read_exact(len_buf).unwrap();
        let msg_len = u32::from_le_bytes(*len_buf) as usize;
        let mut msg_buf = ::alloc::vec::from_elem(0u8, msg_len);
        stream.read_exact(&mut msg_buf).unwrap();
        Response::deserialize(&msg_buf)
    }
    #[inline]
    fn send_message_to_channel(remote: &RemoteBackend, msg: Response) {
        let task_id = msg.task_id;
        let sender = {
            let mut pending = remote.pending_response.write().unwrap();
            pending.remove(&task_id)
        };
        if let Some(sender) = sender {
            sender.send(msg.message).unwrap();
        }
    }
    /// Thread function to drain outgoing messages and send them over the TCP stream
    /// # Arguments
    /// * `remote` - The RemoteBackend instance
    /// * `stream` - The TCP stream to send messages over
    fn drain_outgoing(remote: RemoteBackend, mut stream: std::net::TcpStream) {
        let receiver = remote.messages_outgoing_receiver.clone();
        loop {
            if let Ok(req) = receiver.recv() {
                if remote.is_poisoned() {
                    break;
                }
                let serialized = req.serialize().unwrap();
                let n = serialized.len();
                let n_bytes = (n as u32).to_le_bytes();
                stream.write_all(&n_bytes).unwrap();
                stream.write_all(&serialized).unwrap();
            } else {
                break;
            }
        }
    }
    /// Thread function to read incoming messages from the TCP stream
    ///
    /// Upon receiving a message, if it is marked as not asynchronous, it sends the message to the waiting channel
    /// Otherwise, it handles asynchronous messages accordingly. If the asynchronous message is complete and does not
    /// indicate an error, it simply decrements the pending count. If it indicates an error, it poisons the RemoteBackend to prevent further operations.
    /// If the asynchronous message is not complete, it sends a follow-up message to the waiting channel, and does not decrement the pending count.
    ///
    /// # Arguments
    /// * `remote` - The RemoteBackend instance
    /// * `stream` - The TCP stream to read messages from
    fn read_incoming(remote: RemoteBackend, mut stream: std::net::TcpStream) {
        let mut len_buf = [0u8; 4];
        loop {
            let msg = read_response(&mut stream, &mut len_buf).unwrap();
            if !msg.asynchronous {
                if true {
                    if !msg.complete {
                        ::core::panicking::panic("assertion failed: msg.complete")
                    }
                }
                send_message_to_channel(&remote, msg);
                remote.pending.dec();
            } else if msg.complete {
                if let Some(e) = msg.error {
                    remote.poison();
                    {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "Inconsistent state detected. Received error in async message: {0:?}",
                                e,
                            ),
                        );
                    };
                }
                remote.pending.dec();
            } else {
                send_message_to_channel(&remote, msg);
            }
        }
    }
}
