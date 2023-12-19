use tch::{
    nn, nn::embedding, nn::layer_norm, nn::linear, nn::Module, nn::Path, nn::VarStore, Device,
    IndexOp, Tensor,
};

#[derive(Debug)]
pub struct ChessModel {
    em_board: nn::Embedding,
    em_piece: nn::Embedding,
    f1: nn::Linear,
    f2: nn::Linear,
    f3: nn::Linear,
    f4: nn::Linear,
    f5: nn::Linear,
    layer_norm: nn::LayerNorm,
}

impl ChessModel {
    pub fn new(vs: &Path) -> ChessModel {
        ChessModel {
            em_board: embedding(vs / "em_board", 13, 16, Default::default()),
            em_piece: embedding(vs / "em_piece", 64, 64, Default::default()),
            f1: linear(vs / "f1", 1088, 620, Default::default()),
            f2: linear(vs / "f2", 620, 620, Default::default()),
            f3: linear(vs / "f3", 620, 620, Default::default()),
            f4: linear(vs / "f4", 620, 620, Default::default()),
            f5: linear(vs / "f5", 620, 64, Default::default()),
            layer_norm: layer_norm(vs / "layer_norm", Vec::from([620]), Default::default()),
        }
    }

    pub fn load() -> Option<ChessModel> {
        let mut vs = VarStore::new(Device::cuda_if_available());
        let model = ChessModel::new(&vs.root());

        if let Err(msg) = vs.load("model/chess.safetensors") {
            println!("Failed to load ChessModel! {:?}", msg);
            return None;
        }

        Some(model)
    }

    pub fn prep_input(&self, boards: &[[i32; 64]], pieces: &[i32]) -> Tensor {
        Tensor::concat(
            &[
                self.em_board
                    .forward(
                        &Tensor::from_slice2(boards)
                            .repeat([pieces.len() as i64, 1])
                            .to_device(Device::cuda_if_available()),
                    )
                    .flatten(1, -1),
                self.em_piece
                    .forward(&Tensor::from_slice(pieces).to_device(Device::cuda_if_available())),
            ],
            1,
        )
    }

    pub fn output_to_vec(out: &Tensor) -> Vec<Vec<f64>> {
        (0..out.size()[0])
            .map(|i| {
                (0..64)
                    .map(|i2| out.i(i).double_value(&[i2]))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }
}

impl Module for ChessModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut data = self
            .layer_norm
            .forward(&self.f1.forward(input).gelu("none"));
        data = self
            .layer_norm
            .forward(&self.f2.forward(&data).gelu("none"))
            + data;
        data = self
            .layer_norm
            .forward(&self.f3.forward(&data).gelu("none"))
            + data;
        data = self
            .layer_norm
            .forward(&self.f4.forward(&data).gelu("none"))
            + data;

        self.f5.forward(&data)
    }
}
