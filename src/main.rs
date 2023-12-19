#![feature(portable_simd)]

mod chessbot;
mod chessfrommodel;
mod chessmodel;
mod evaluation;
mod mlchess;

use crate::chessbot::ChessBot;
use crate::chessfrommodel::ChessFromModel;
use crate::chessmodel::ChessModel;
use crate::evaluation::eval_pos;
use crate::mlchess::{best_from, encode_board};
use chess::{Board, ChessMove};
use std::io;
use std::str::FromStr;
use tch::{nn::Module, Device, Tensor};

fn process_complex_cmd(bot: &mut ChessBot, cmd: &str) {
    let len = cmd.len();

    if len >= 18 && len < 24 && &cmd[0..17] == "position startpos" {
        bot.reset_board();
    } else if len > 8 && &cmd[0..8] == "position" {
        let pieces = cmd[..len - 1].split(' ').collect::<Vec<&str>>()[3..].to_vec();

        bot.reset_board();

        for m in &pieces {
            let mv = ChessMove::from_str(m).expect("A legal move");
            bot.make_move(mv);
        }
    } else if len >= 2 && &cmd[0..2] == "go" {
        if let Some(res) = bot.find_best_move(1) {
            if let Some(mv) = res.0 {
                println!(
                    "bestmove {}{}",
                    ChessBot::sqr_to_str(mv.get_source().to_index() as i32),
                    ChessBot::sqr_to_str(mv.get_dest().to_index() as i32)
                );
            }
        }
    }
}

fn run(bot: &mut ChessBot) {
    let mut input = String::new();
    
    if io::stdin().read_line(&mut input).is_err() {
        println!("info string Warning: Cannot process input")
    }
    match input.as_str() {
        "uci\n" => {
            println!("id name NightyBot\nid author Nighty");
            println!("option name Depth type spin default 6 min 1 max 32");
            println!("option name AnalyzeMoves type spin default 2 min 1 max 24");
            println!("option name AnalyzePieces type spin default 3 min 1 max 16");
            println!("option name Move Overhead type spin default 100 min 1 max 1000");
            println!("option name Threads type spin default 1 min 1 max 64");
            println!("option name Hash type spin default 4096 min 1 max 999999999");
            println!("option name SyzygyPath type string default");
            println!("option name UCI_ShowWDL type check default true");
            println!("uciok");
        }
        "isready\n" => {
            println!("readyok");
        }
        "ucinewgame\n" => bot.reset_board(),
        "quit\n" => {
            std::process::exit(0);
        }
        "benchmark\n" => {
            benchmark();
        }
        "test\n" => {
            test();
        }
        cmd => {
            process_complex_cmd(bot, cmd);
        }
    }
    run(bot);
}

fn test() {
    let board = Board::default();
    let model = ChessModel::load().unwrap();
    let fmodel = ChessFromModel::load().unwrap();
    let piece: i32 = 8;
    let input = model
        .prep_input(
            &vec![encode_board(&board), encode_board(&board)],
            &[piece, piece],
        )
        .to_device(Device::cuda_if_available());

    println!("{:?}", ChessModel::output_to_vec(&model.forward(&input)));
    println!("{:?}", best_from(&fmodel, &[&board, &board]))
}

fn benchmark() {
    let board = Board::default();
    let mut encoding: [i32; 64] = [0; 64];
    let mut eval = 0;

    use std::time::Instant;

    let now = Instant::now();

    for _i in 0..1000000 {
        encoding = encode_board(&board);
    }

    let s = now.elapsed().as_secs() as f32;
    let ns = now.elapsed().subsec_nanos() as f32;
    let total = s + ns / 1_000_000_000.0;

    println!(
        "Encoded 1M boards in {:.2?} (that's {:.0?} encodings/sec) {:?}",
        now.elapsed(),
        1000000.0 / total,
        encoding
    );

    let now = Instant::now();

    for _i in 0..1000000 {
        eval = eval_pos(&board, 0);
    }

    let s = now.elapsed().as_secs() as f32;
    let ns = now.elapsed().subsec_nanos() as f32;
    let total = s + ns / 1_000_000_000.0;

    println!(
        "Evaluated 1M boards in {:.2?} (that's {:.0?} evaluations/sec) {:?}",
        now.elapsed(),
        1000000.0 / total,
        eval
    );

    let now = Instant::now();

    for _i in 0..1000000 {
        encoding = encode_board(&board);
        eval = eval_pos(&board, 0);
    }

    let s = now.elapsed().as_secs() as f32;
    let ns = now.elapsed().subsec_nanos() as f32;
    let total = s + ns / 1_000_000_000.0;

    println!(
        "Encoded & Evaluated 1M boards in {:.2?} (that's {:.0?} boards/sec) {:?} {:?}",
        now.elapsed(),
        1000000.0 / total,
        eval,
        encoding
    );

    let model = ChessFromModel::load().unwrap();
    let input = Tensor::from_slice2(&[encode_board(&board), encode_board(&board)])
        .to_device(Device::cuda_if_available());
    let mut out = model.forward(&input);
    let now = Instant::now();

    for _i in 0..10000 {
        out = model.forward(&input);
    }

    let s = now.elapsed().as_secs() as f32;
    let ns = now.elapsed().subsec_nanos() as f32;
    let total = s + ns / 1_000_000_000.0;

    println!(
        "Ran ChessFromModel 10000 times in {:.2?} (that's {:.0?} executions/sec) {}",
        now.elapsed(),
        10000.0 / total,
        out
    );

    let model = ChessModel::load().unwrap();
    let piece: i32 = 8;
    let input = model
        .prep_input(
            &vec![encode_board(&board), encode_board(&board)],
            &[piece, piece],
        )
        .to_device(Device::cuda_if_available());
    let mut out = model.forward(&input);
    let now = Instant::now();

    for _i in 0..10000 {
        out = model.forward(&input);
    }

    let s = now.elapsed().as_secs() as f32;
    let ns = now.elapsed().subsec_nanos() as f32;
    let total = s + ns / 1_000_000_000.0;

    println!(
        "Ran ChessModel 10000 times in {:.2?} (that's {:.0?} executions/sec) {}",
        now.elapsed(),
        10000.0 / total,
        out
    );
}

fn main() {
    let mut bot = ChessBot::new();

    run(&mut bot);
}
