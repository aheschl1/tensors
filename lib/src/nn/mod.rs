use slotmap::{new_key_type, SlotMap};
use thiserror::Error;

use crate::{backend::{Backend, BackendMatMul}, core::{primitives::{DeviceType, TensorBase, TensorId}, value::WeightValue}, ops::{linalg::MatMul, unary::UnaryOp}};
#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};


#[derive(Debug, Error, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum GraphError {
    #[error("wrong number of inputs {0} for {1}")]
    WrongNumberOfInputs(usize, &'static str),

    #[error("missing tensor")]
    MissingTensor,

    #[error("input already set for node")]
    InputAlreadySet,

    #[error("tensor error: {0}")]
    TensorError(#[from] crate::core::tensor::TensorError),
}

new_key_type! {
    pub struct NodeId;
}

pub mod graph {
    use crate::{backend::{cpu::Cpu, cuda::Cuda, BackendMatMul}, core::{primitives::{TensorBase, TensorId}, value::WeightValue}, nn::{GraphContext, Node}};

    pub enum Contexts {
        #[cfg(feature = "cuda")]
        CudaF32(GraphContext<f32, Cuda>),
        CpuF32(GraphContext<f32, Cpu>),
    }

    #[macro_export]
    macro_rules! create_context {
        ($device:tt, $dtype:tt) => {
            match ($device, $dtype) {
                ("cpu", "f32") => {
                    $crate::nn::graph::Contexts::CpuF32($crate::nn::GraphContext::<f32, $crate::backend::cpu::Cpu>::new())
                },
                #[cfg(feature = "cuda")]
                ("cuda", "f32") => {
                    $crate::nn::graph::Contexts::CudaF32($crate::nn::GraphContext::<f32, $crate::backend::cuda::Cuda>::new())
                },
                _ => panic!("Unsupported device or dtype"),
            }
        };
    }

    // pass through to the context inner
    impl Contexts {
        pub fn add_node(&mut self, node: Node) {
            match self {
                Contexts::CudaF32(ctx) => ctx.add_node(node),
                Contexts::CpuF32(ctx) => ctx.add_node(node),
            }
        }

        pub fn register_tensor<T: WeightValue, B: BackendMatMul<T>>(&mut self, tensor: TensorBase<T, B>) -> &mut TensorBase<T, B> {
            let tensor_any = Box::new(tensor) as Box<dyn std::any::Any>;
            // TODO this is wildly unsafe
            // it is actually fucking terrible
            match self {
                #[cfg(feature = "cuda")]
                Contexts::CudaF32(ctx) => {
                    let boxed = tensor_any.downcast::<TensorBase<f32, Cuda>>().expect("Expected TensorBase<f32, Cuda>");
                    let tensor_ref = ctx.register_tensor(*boxed);
                    unsafe { std::mem::transmute::<&mut TensorBase<f32, Cuda>, &mut TensorBase<T, B>>(tensor_ref) }
                },
                Contexts::CpuF32(ctx) => {
                    let boxed = tensor_any.downcast::<TensorBase<f32, Cpu>>().expect("Expected TensorBase<f32, Cpu>");
                    let tensor_ref = ctx.register_tensor(*boxed);
                    unsafe { std::mem::transmute::<&mut TensorBase<f32, Cpu>, &mut TensorBase<T, B>>(tensor_ref) }
                },
            }
        }
    }

    thread_local! {
        static GRAPH_CONTEXT: std::cell::RefCell<Option<Contexts>> = std::cell::RefCell::new(None); 
    }

    pub fn init(ctx: Contexts) {
        GRAPH_CONTEXT.with(|ctx_cell| {
            let mut ctx_ref = ctx_cell.borrow_mut();
            *ctx_ref = Some(ctx);
        });
    }

    pub fn with<R>(f: impl FnOnce(&mut Contexts) -> R) -> R {
        // initialize context if needed
        GRAPH_CONTEXT.with(|ctx_cell| {
            let mut ctx_ref = ctx_cell.borrow_mut();
            let ctx = ctx_ref.as_mut().expect("Graph context not initialized");
            f(ctx)
        })
    }
}

pub struct ExecCtx<'a, T: WeightValue, B: BackendMatMul<T>> {
    backend: &'a B,
    tensors: SlotMap<TensorId, TensorBase<T, B>>,
}

impl<'a, T: WeightValue, B: BackendMatMul<T>> ExecCtx<'a, T, B> {
    pub fn new(backend: &'a B) -> Self {
        Self {
            backend,
            tensors: SlotMap::with_key(),
        }
    }

    pub fn tensor(&self, id: TensorId) -> Option<&TensorBase<T, B>> {
        self.tensors.get(id)
    }

    pub fn add_tensor(&mut self, tensor: TensorBase<T, B>) -> TensorId {
        self.tensors.insert(tensor)
    }
}

pub enum Node {
    MatMul {
        lhs: TensorId,
        rhs: TensorId,
        output: TensorId
    }
}

pub struct LinearLayer<T: WeightValue, B: BackendMatMul<T>> {
    weights: TensorId,
    bias: Option<TensorId>,
    input: Option<TensorId>,
    _marker: std::marker::PhantomData<(B, T)>,
}

impl<T: WeightValue, B: BackendMatMul<T>> LinearLayer<T, B> {
    pub fn forward(&mut self, inputs: &Vec<TensorId>, ctx: &mut ExecCtx<T, B>) -> Result<TensorId, GraphError> {
        if inputs.len() != 1 {
            return Err(GraphError::WrongNumberOfInputs(inputs.len(), "LinearLayer"));
        }
        if self.input.is_some() {
            return Err(GraphError::InputAlreadySet);
        }
        self.input = Some(inputs[0]);

        let input_tensor = ctx.tensor(inputs[0]).ok_or(GraphError::MissingTensor)?;
        let weight_tensor = ctx.tensor(self.weights).ok_or(GraphError::MissingTensor)?;
        let output_tensor = input_tensor.matmul(weight_tensor)?;
        Ok(ctx.add_tensor(output_tensor))
    }

    pub fn backward(&mut self) -> TensorBase<T, B> {
        // return upstream gradient, and save gradients for weights and bias
        todo!()
    }
}

pub struct ReLU<T: WeightValue, B: BackendMatMul<T>> {
    _marker: std::marker::PhantomData<(B, T)>,
}

impl<T: WeightValue, B: BackendMatMul<T>> ReLU<T, B> {
    pub fn forward(&self, inputs: &Vec<TensorId>, ctx: &mut ExecCtx<T, B>) -> Result<TensorId, GraphError> {
        if inputs.len() != 1 {
            return Err(GraphError::WrongNumberOfInputs(inputs.len(), "ReLU"));
        }
        let input_tensor = ctx.tensor(inputs[0]).ok_or(GraphError::MissingTensor)?;
        let output_tensor = input_tensor.relu();
        Ok(ctx.add_tensor(output_tensor))
    } 
}

pub struct GraphContext<T: WeightValue, B: BackendMatMul<T>> {
    last_node: Option<NodeId>,
    graph: Graph,
    tensors: SlotMap<TensorId, TensorBase<T, B>>,
}

impl<T: WeightValue, B: BackendMatMul<T>> GraphContext<T, B> {
    pub fn new() -> Self {
        Self {
            last_node: None,
            tensors: SlotMap::with_key(),
            graph: Graph {
                nodes: SlotMap::with_key(),
                edges: Vec::new(),
                head: None,
            },
        }
    }

    pub fn add_node(&mut self, node: Node) {
        let node_id = self.graph.nodes.insert(node);
        if let Some(last_node_id) = self.last_node {
            self.graph.edges.push((last_node_id, node_id));
        }
        self.last_node = Some(node_id);
    }

    pub fn register_tensor(&mut self, tensor: TensorBase<T, B>) -> &mut TensorBase<T, B> {
        let id = self.tensors.insert(tensor);
        let t = self.tensors.get_mut(id).unwrap();
        t.id = Some(id);
        t
    }
}

pub struct Graph {
    nodes: SlotMap<NodeId, Node>,
    edges: Vec<(NodeId, NodeId)>,
    head: Option<NodeId>,
}

#[cfg(test)]
mod tests {
    use crate::{backend::{cpu::Cpu, cuda::Cuda}, core::Tensor, create_context};

    use super::*;
    #[test]
    fn test_graph_stuff(){
        let context = create_context!("cpu", "f32");
        graph::init(context);
        graph::with(|ctx| {
            let tensor = Tensor::<f32>::ones((1, 1));
            let tensor = ctx.register_tensor(tensor);
            let tb = tensor + 1.0;
        })
    }
}