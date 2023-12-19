use tch::{
    nn, nn::embedding, nn::layer_norm, nn::linear, nn::Module, nn::Path, nn::VarStore, Device,
    IndexOp, Tensor,
};

#[derive(Debug)]
pub struct ChessFromModel {
    em_board: nn::Embedding,
    f1: nn::Linear,
    f2: nn::Linear,
    f3: nn::Linear,
    f4: nn::Linear,
    f5: nn::Linear,
    layer_norm: nn::LayerNorm,
}

impl ChessFromModel {
    pub fn new(vs: &Path) -> ChessFromModel {
        ChessFromModel {
            em_board: embedding(vs / "em_board", 13, 16, Default::default()),
            f1: linear(vs / "f1", 1024, 620, Default::default()),
            f2: linear(vs / "f2", 620, 620, Default::default()),
            f3: linear(vs / "f3", 620, 620, Default::default()),
            f4: linear(vs / "f4", 620, 620, Default::default()),
            f5: linear(vs / "f5", 620, 128, Default::default()),
            layer_norm: layer_norm(vs / "layer_norm", Vec::from([620]), Default::default()),
        }
    }

    pub fn load() -> Option<ChessFromModel> {
        let mut vs = VarStore::new(Device::cuda_if_available());
        let model = ChessFromModel::new(&vs.root());

        if let Err(msg) = vs.load("model/chess_from.safetensors") {
            println!("Failed to load ChessFromModel! {:?}", msg);
            return None;
        }

        Some(model)
    }

    pub fn output_to_vec(out: &Tensor) -> Vec<Vec<f64>> {
        (0..out.size()[0])
            .map(|i| {
                (0..128)
                    .map(|i2| out.i(i).double_value(&[i2]))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }
}

impl Module for ChessFromModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut data = self.em_board.forward(input).flatten(1, -1);
        data = self
            .layer_norm
            .forward(&self.f1.forward(&data).gelu("none"));
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
