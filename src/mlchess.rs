use crate::chessfrommodel::ChessFromModel;
use crate::chessmodel::ChessModel;
use chess::{Board, Color};
use tch::{nn::Module, Device, Tensor};

fn swap_piece(piece: i32) -> i32 {
    match piece {
        3 => 1,
        2 => 3,
        1 => 2,
        _ => piece,
    }
}

pub fn encode_board(board: &Board) -> [i32; 64] {
    let mut out: [i32; 64] = [0; 64];

    for sqr in board.color_combined(Color::White).into_iter() {
        if let Some(piece) = board.piece_on(sqr) {
            out[sqr.to_index()] = 1 + swap_piece(piece as i32);
        }
    }

    for sqr in board.color_combined(Color::Black).into_iter() {
        if let Some(piece) = board.piece_on(sqr) {
            out[sqr.to_index()] = 7 + swap_piece(piece as i32);
        }
    }

    out
}

pub fn best_from(model: &ChessFromModel, boards: &[&Board]) -> Vec<[Vec<usize>; 2]> {
    ChessFromModel::output_to_vec(
        &model.forward(
            &Tensor::from_slice2(
                &boards
                    .iter()
                    .map(|b| encode_board(b))
                    .collect::<Vec<[i32; 64]>>(),
            )
            .to_device(Device::cuda_if_available()),
        ),
    )
    .iter()
    .map(|r| {
        let mut white: Vec<(usize, f64)> =
            r[0..64].iter().enumerate().map(|s| (s.0, *s.1)).collect();
        let mut black: Vec<(usize, f64)> =
            r[64..128].iter().enumerate().map(|s| (s.0, *s.1)).collect();

        white.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
        black.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());

        [
            white.iter().map(|o| o.0).collect(),
            black.iter().map(|o| o.0).collect(),
        ]
    })
    .collect()
}

pub fn best_to(model: &ChessModel, boards: &[&Board], squares: &[usize]) -> Vec<Vec<usize>> {
    ChessModel::output_to_vec(
        &model.forward(
            &model.prep_input(
                &boards
                    .iter()
                    .map(|b| encode_board(b))
                    .collect::<Vec<[i32; 64]>>(),
                &squares.iter().map(|s| *s as i32).collect::<Vec<i32>>(),
            ),
        ),
    )
    .iter()
    .map(|r| {
        let mut pieces: Vec<(usize, f64)> = r.iter().enumerate().map(|s| (s.0, *s.1)).collect();

        pieces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());

        pieces.iter().map(|o| o.0).collect()
    })
    .collect()
}
